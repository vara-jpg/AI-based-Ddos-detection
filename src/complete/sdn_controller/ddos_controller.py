#!/usr/bin/env python3
"""
AI-Driven Low-Rate DDoS Detection SDN Controller
Main Ryu controller for monitoring network traffic and detecting DDoS attacks
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime
import yaml
import logging

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.topology import event as topo_event
from ryu.topology.api import get_switch, get_link

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.flow_analyzer import FlowAnalyzer
from utils.logger_config import setup_logger
from detection.attack_detector import AttackDetector


class DDoSDetectionController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(DDoSDetectionController, self).__init__(*args, **kwargs)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize logging
        self.logger = setup_logger('SDN_Controller', self.config['logging'])
        
        # Initialize components
        self.flow_analyzer = FlowAnalyzer(self.config)
        self.attack_detector = AttackDetector(self.config, self.logger)
        
        # Data structures
        self.mac_to_port = {}
        self.switches = {}
        self.flow_stats = defaultdict(lambda: deque(maxlen=100))
        self.port_stats = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_network)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("DDoS Detection Controller initialized")
    
    def _load_config(self):
        """Load SDN controller configuration"""
        try:
            with open('config/sdn_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration if file not found"""
        return {
            'controller': {'port': 6633, 'host': '0.0.0.0'},
            'monitoring': {'flow_stats_interval': 1, 'port_stats_interval': 1},
            'flow_table': {'idle_timeout': 60, 'hard_timeout': 300, 'priority_base': 100},
            'mitigation': {'enable_auto_mitigation': True},
            'logging': {'level': 'INFO', 'log_file': 'logs/sdn_controller.log'}
        }
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        self.switches[datapath.id] = datapath
        self.logger.info(f"Switch {datapath.id} connected")
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Add flow entry to switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Avoid broadcast from LLDP
        if eth_pkt.ethertype == 35020:
            return
        
        dst = eth_pkt.dst
        src = eth_pkt.src
        dpid = datapath.id
        
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn MAC address
        self.mac_to_port[dpid][src] = in_port
        
        # Analyze packet for DDoS detection
        self._analyze_packet(pkt, dpid, src, dst, in_port)
        
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Check for potential attack before installing flow
            if self._check_flow_safety(src, dst, dpid):
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _analyze_packet(self, pkt, dpid, src, dst, in_port):
        """Analyze packet for attack patterns"""
        timestamp = time.time()
        
        # Extract packet features
        features = self.flow_analyzer.extract_packet_features(pkt, timestamp)
        
        # Update flow statistics
        flow_key = f"{src}-{dst}"
        self.flow_stats[flow_key].append({
            'timestamp': timestamp,
            'dpid': dpid,
            'in_port': in_port,
            'features': features
        })
        
        # Trigger detection if enough data collected
        if len(self.flow_stats[flow_key]) >= 10:
            self._trigger_detection(flow_key)
    
    def _check_flow_safety(self, src, dst, dpid):
        """Check if flow is safe to install"""
        flow_key = f"{src}-{dst}"
        
        # Get recent attack predictions
        recent_predictions = self.attack_detector.get_recent_predictions(flow_key)
        
        if recent_predictions:
            attack_probability = max(recent_predictions)
            if attack_probability > 0.7:  # High attack probability
                self.logger.warning(f"Blocking flow installation for {flow_key} due to high attack probability: {attack_probability}")
                return False
        
        return True
    
    def _trigger_detection(self, flow_key):
        """Trigger attack detection for a flow"""
        try:
            flow_data = list(self.flow_stats[flow_key])
            prediction = self.attack_detector.detect_attack(flow_data)
            
            if prediction['is_attack']:
                self._handle_attack_detection(flow_key, prediction)
                
        except Exception as e:
            self.logger.error(f"Error in attack detection: {e}")
    
    def _handle_attack_detection(self, flow_key, prediction):
        """Handle detected attack"""
        self.logger.critical(f"ATTACK DETECTED: {flow_key}")
        self.logger.critical(f"Attack probability: {prediction['probability']:.3f}")
        self.logger.critical(f"Attack type: {prediction.get('attack_type', 'Unknown')}")
        
        # Apply mitigation if enabled
        if self.config['mitigation']['enable_auto_mitigation']:
            self._apply_mitigation(flow_key, prediction)
    
    def _apply_mitigation(self, flow_key, prediction):
        """Apply mitigation strategies"""
        src, dst = flow_key.split('-')
        
        # Rate limiting
        self._apply_rate_limiting(src, dst)
        
        # Log mitigation action
        self.logger.info(f"Applied mitigation for flow {flow_key}")
    
    def _apply_rate_limiting(self, src, dst):
        """Apply rate limiting to suspicious flow"""
        for dpid, datapath in self.switches.items():
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            
            # Create match for the flow
            match = parser.OFPMatch(eth_src=src, eth_dst=dst)
            
            # Create meter for rate limiting
            bands = [parser.OFPMeterBandDrop(rate=1000, burst_size=100)]
            meter_mod = parser.OFPMeterMod(datapath=datapath, command=ofproto.OFPMC_ADD,
                                         flags=ofproto.OFPMF_PKTPS, meter_id=1, bands=bands)
            datapath.send_msg(meter_mod)
            
            # Install flow with meter
            inst = [parser.OFPInstructionMeter(meter_id=1),
                   parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                              [parser.OFPActionOutput(ofproto.OFPP_NORMAL)])]
            
            flow_mod = parser.OFPFlowMod(datapath=datapath, priority=1000,
                                       match=match, instructions=inst)
            datapath.send_msg(flow_mod)
    
    def _monitor_network(self):
        """Monitor network statistics"""
        while self.monitoring_active:
            try:
                self._request_flow_stats()
                self._request_port_stats()
                time.sleep(self.config['monitoring']['flow_stats_interval'])
            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")
    
    def _request_flow_stats(self):
        """Request flow statistics from all switches"""
        for datapath in self.switches.values():
            parser = datapath.ofproto_parser
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)
    
    def _request_port_stats(self):
        """Request port statistics from all switches"""
        for datapath in self.switches.values():
            parser = datapath.ofproto_parser
            req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
            datapath.send_msg(req)
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply"""
        flows = []
        for stat in ev.msg.body:
            flows.append({
                'table_id': stat.table_id,
                'match': stat.match,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration': stat.duration_sec
            })
        
        # Store flow statistics
        dpid = ev.msg.datapath.id
        self.flow_stats[f"switch_{dpid}"].append({
            'timestamp': time.time(),
            'flows': flows
        })
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply"""
        ports = []
        for stat in ev.msg.body:
            ports.append({
                'port_no': stat.port_no,
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors
            })
        
        # Store port statistics
        dpid = ev.msg.datapath.id
        self.port_stats[f"switch_{dpid}"].append({
            'timestamp': time.time(),
            'ports': ports
        })
    
    def stop(self):
        """Stop the controller"""
        self.monitoring_active = False
        self.logger.info("DDoS Detection Controller stopped")


if __name__ == '__main__':
    from ryu.cmd import manager
    import sys
    
    # Run the controller
    sys.argv.extend(['--ofp-tcp-listen-port', '6633'])
    manager.main()
