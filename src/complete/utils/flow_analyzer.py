#!/usr/bin/env python3
"""
Flow Analyzer for Network Traffic
Extracts features from network packets and flows for DDoS detection
"""

import numpy as np
import time
from collections import defaultdict, deque
from typing import Dict, List, Any
import math
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp


class FlowAnalyzer:
    """
    Analyzes network flows to extract features for DDoS detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Flow Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.flow_cache = defaultdict(lambda: deque(maxlen=1000))
        self.flow_stats = defaultdict(dict)
        
        # Time windows for analysis
        self.time_window = 10  # seconds
        self.feature_cache = defaultdict(lambda: deque(maxlen=100))
        
    def extract_packet_features(self, pkt: packet.Packet, timestamp: float) -> Dict[str, float]:
        """
        Extract features from a single packet
        
        Args:
            pkt: Network packet
            timestamp: Packet timestamp
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic packet information
        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        features['packet_size'] = len(pkt.data) if hasattr(pkt, 'data') else 0
        features['timestamp'] = timestamp
        
        # Protocol analysis
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        if ipv4_pkt:
            features.update(self._extract_ip_features(ipv4_pkt))
            
            # TCP analysis
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            if tcp_pkt:
                features.update(self._extract_tcp_features(tcp_pkt))
                features['protocol_type'] = 6  # TCP
            
            # UDP analysis
            udp_pkt = pkt.get_protocol(udp.udp)
            if udp_pkt:
                features.update(self._extract_udp_features(udp_pkt))
                features['protocol_type'] = 17  # UDP
            
            # ICMP analysis
            icmp_pkt = pkt.get_protocol(icmp.icmp)
            if icmp_pkt:
                features.update(self._extract_icmp_features(icmp_pkt))
                features['protocol_type'] = 1  # ICMP
        else:
            features['protocol_type'] = 0  # Other
        
        return features
    
    def _extract_ip_features(self, ip_pkt: ipv4.ipv4) -> Dict[str, float]:
        """Extract features from IP packet"""
        features = {}
        
        features['ip_total_length'] = ip_pkt.total_length
        features['ip_header_length'] = ip_pkt.header_length
        features['ip_tos'] = ip_pkt.tos
        features['ip_identification'] = ip_pkt.identification
        features['ip_flags'] = ip_pkt.flags
        features['ip_offset'] = ip_pkt.offset
        features['ip_ttl'] = ip_pkt.ttl
        features['ip_proto'] = ip_pkt.proto
        
        # Convert IP addresses to numerical features
        src_ip = self._ip_to_int(ip_pkt.src)
        dst_ip = self._ip_to_int(ip_pkt.dst)
        
        features['src_ip_last_octet'] = src_ip & 0xFF
        features['dst_ip_last_octet'] = dst_ip & 0xFF
        
        return features
    
    def _extract_tcp_features(self, tcp_pkt: tcp.tcp) -> Dict[str, float]:
        """Extract features from TCP packet"""
        features = {}
        
        features['src_port'] = tcp_pkt.src_port
        features['dst_port'] = tcp_pkt.dst_port
        features['tcp_seq'] = tcp_pkt.seq % 1000000  # Normalize
        features['tcp_ack'] = tcp_pkt.ack % 1000000  # Normalize
        features['tcp_offset'] = tcp_pkt.offset
        features['tcp_bits'] = tcp_pkt.bits
        features['tcp_window_size'] = tcp_pkt.window_size
        features['tcp_csum'] = tcp_pkt.csum
        features['tcp_urgent'] = tcp_pkt.urgent
        
        # TCP flag analysis
        features['tcp_fin'] = 1 if tcp_pkt.bits & 0x01 else 0
        features['tcp_syn'] = 1 if tcp_pkt.bits & 0x02 else 0
        features['tcp_rst'] = 1 if tcp_pkt.bits & 0x04 else 0
        features['tcp_psh'] = 1 if tcp_pkt.bits & 0x08 else 0
        features['tcp_ack_flag'] = 1 if tcp_pkt.bits & 0x10 else 0
        features['tcp_urg'] = 1 if tcp_pkt.bits & 0x20 else 0
        
        return features
    
    def _extract_udp_features(self, udp_pkt: udp.udp) -> Dict[str, float]:
        """Extract features from UDP packet"""
        features = {}
        
        features['src_port'] = udp_pkt.src_port
        features['dst_port'] = udp_pkt.dst_port
        features['udp_total_length'] = udp_pkt.total_length
        features['udp_csum'] = udp_pkt.csum
        
        return features
    
    def _extract_icmp_features(self, icmp_pkt: icmp.icmp) -> Dict[str, float]:
        """Extract features from ICMP packet"""
        features = {}
        
        features['icmp_type'] = icmp_pkt.type
        features['icmp_code'] = icmp_pkt.code
        features['icmp_csum'] = icmp_pkt.csum
        features['icmp_data_len'] = len(icmp_pkt.data) if icmp_pkt.data else 0
        
        return features
    
    def extract_flow_features(self, flow_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract statistical features from a flow
        
        Args:
            flow_data: List of packet data for the flow
            
        Returns:
            Dictionary of flow-level features
        """
        if not flow_data:
            return {}
        
        features = {}
        
        # Basic flow statistics
        features['packet_count'] = len(flow_data)
        features['flow_duration'] = flow_data[-1]['timestamp'] - flow_data[0]['timestamp']
        
        # Packet size statistics
        packet_sizes = [d['features'].get('packet_size', 0) for d in flow_data]
        features['total_bytes'] = sum(packet_sizes)
        features['avg_packet_size'] = np.mean(packet_sizes) if packet_sizes else 0
        features['std_packet_size'] = np.std(packet_sizes) if packet_sizes else 0
        features['min_packet_size'] = min(packet_sizes) if packet_sizes else 0
        features['max_packet_size'] = max(packet_sizes) if packet_sizes else 0
        
        # Temporal features
        if len(flow_data) > 1:
            inter_arrival_times = []
            for i in range(1, len(flow_data)):
                iat = flow_data[i]['timestamp'] - flow_data[i-1]['timestamp']
                inter_arrival_times.append(iat)
            
            if inter_arrival_times:
                features['avg_inter_arrival_time'] = np.mean(inter_arrival_times)
                features['std_inter_arrival_time'] = np.std(inter_arrival_times)
                features['min_inter_arrival_time'] = min(inter_arrival_times)
                features['max_inter_arrival_time'] = max(inter_arrival_times)
        else:
            features['avg_inter_arrival_time'] = 0
            features['std_inter_arrival_time'] = 0
            features['min_inter_arrival_time'] = 0
            features['max_inter_arrival_time'] = 0
        
        # Rate features
        if features['flow_duration'] > 0:
            features['packets_per_second'] = features['packet_count'] / features['flow_duration']
            features['bytes_per_second'] = features['total_bytes'] / features['flow_duration']
        else:
            features['packets_per_second'] = 0
            features['bytes_per_second'] = 0
        
        # Protocol distribution
        protocol_counts = defaultdict(int)
        for d in flow_data:
            proto = d['features'].get('protocol_type', 0)
            protocol_counts[proto] += 1
        
        features['tcp_ratio'] = protocol_counts[6] / len(flow_data) if flow_data else 0
        features['udp_ratio'] = protocol_counts[17] / len(flow_data) if flow_data else 0
        features['icmp_ratio'] = protocol_counts[1] / len(flow_data) if flow_data else 0
        
        # Port diversity
        src_ports = set()
        dst_ports = set()
        for d in flow_data:
            src_port = d['features'].get('src_port', 0)
            dst_port = d['features'].get('dst_port', 0)
            if src_port:
                src_ports.add(src_port)
            if dst_port:
                dst_ports.add(dst_port)
        
        features['unique_src_ports'] = len(src_ports)
        features['unique_dst_ports'] = len(dst_ports)
        features['src_port_entropy'] = self._calculate_entropy(src_ports) if src_ports else 0
        features['dst_port_entropy'] = self._calculate_entropy(dst_ports) if dst_ports else 0
        
        # TCP flag analysis
        if protocol_counts[6] > 0:  # TCP packets present
            tcp_flags = defaultdict(int)
            for d in flow_data:
                if d['features'].get('protocol_type') == 6:
                    if d['features'].get('tcp_syn'):
                        tcp_flags['syn'] += 1
                    if d['features'].get('tcp_ack_flag'):
                        tcp_flags['ack'] += 1
                    if d['features'].get('tcp_fin'):
                        tcp_flags['fin'] += 1
                    if d['features'].get('tcp_rst'):
                        tcp_flags['rst'] += 1
            
            features['syn_ratio'] = tcp_flags['syn'] / protocol_counts[6]
            features['ack_ratio'] = tcp_flags['ack'] / protocol_counts[6]
            features['fin_ratio'] = tcp_flags['fin'] / protocol_counts[6]
            features['rst_ratio'] = tcp_flags['rst'] / protocol_counts[6]
        else:
            features['syn_ratio'] = 0
            features['ack_ratio'] = 0
            features['fin_ratio'] = 0
            features['rst_ratio'] = 0
        
        return features
    
    def extract_window_features(self, flow_key: str, window_size: int = 10) -> Dict[str, float]:
        """
        Extract features from a time window of flow data
        
        Args:
            flow_key: Flow identifier
            window_size: Time window size in seconds
            
        Returns:
            Dictionary of window-based features
        """
        current_time = time.time()
        window_start = current_time - window_size
        
        # Get packets in the time window
        window_packets = []
        for packet_data in self.flow_cache[flow_key]:
            if packet_data['timestamp'] >= window_start:
                window_packets.append(packet_data)
        
        if not window_packets:
            return {}
        
        # Extract flow features for the window
        flow_features = self.extract_flow_features(window_packets)
        
        # Add window-specific features
        features = flow_features.copy()
        features['window_size'] = window_size
        features['window_packet_count'] = len(window_packets)
        
        # Burst detection
        features['burst_score'] = self._calculate_burst_score(window_packets)
        
        # Regularity score
        features['regularity_score'] = self._calculate_regularity_score(window_packets)
        
        return features
    
    def _calculate_burst_score(self, packets: List[Dict[str, Any]]) -> float:
        """Calculate burst score for packets"""
        if len(packets) < 2:
            return 0.0
        
        timestamps = [p['timestamp'] for p in packets]
        timestamps.sort()
        
        # Calculate variance in inter-arrival times
        inter_arrivals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not inter_arrivals:
            return 0.0
        
        mean_iat = np.mean(inter_arrivals)
        var_iat = np.var(inter_arrivals)
        
        # Burst score is higher when variance is high relative to mean
        if mean_iat > 0:
            return min(var_iat / mean_iat, 10.0)  # Cap at 10
        else:
            return 0.0
    
    def _calculate_regularity_score(self, packets: List[Dict[str, Any]]) -> float:
        """Calculate regularity score for packets"""
        if len(packets) < 3:
            return 0.0
        
        timestamps = [p['timestamp'] for p in packets]
        timestamps.sort()
        
        # Calculate coefficient of variation for inter-arrival times
        inter_arrivals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not inter_arrivals:
            return 0.0
        
        mean_iat = np.mean(inter_arrivals)
        std_iat = np.std(inter_arrivals)
        
        if mean_iat > 0:
            cv = std_iat / mean_iat
            # Regularity is inverse of coefficient of variation
            return max(0.0, 1.0 - cv)
        else:
            return 0.0
    
    def _calculate_entropy(self, values: set) -> float:
        """Calculate entropy of a set of values"""
        if len(values) <= 1:
            return 0.0
        
        # For ports, calculate entropy based on distribution
        total = len(values)
        entropy = 0.0
        
        for value in values:
            p = 1.0 / total  # Assuming uniform distribution
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _ip_to_int(self, ip_str: str) -> int:
        """Convert IP address string to integer"""
        try:
            parts = ip_str.split('.')
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except:
            return 0
    
    def update_flow_cache(self, flow_key: str, packet_data: Dict[str, Any]):
        """Update flow cache with new packet data"""
        self.flow_cache[flow_key].append(packet_data)
    
    def get_flow_features_vector(self, flow_features: Dict[str, float], feature_names: List[str]) -> List[float]:
        """
        Convert flow features dictionary to feature vector
        
        Args:
            flow_features: Dictionary of flow features
            feature_names: List of feature names in desired order
            
        Returns:
            Feature vector as list of floats
        """
        vector = []
        for feature_name in feature_names:
            value = flow_features.get(feature_name, 0.0)
            # Handle potential NaN or infinity values
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(float(value))
        
        return vector
    
    def cleanup_old_data(self, max_age: int = 3600):
        """
        Clean up old data from caches
        
        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        # Clean flow cache
        for flow_key in list(self.flow_cache.keys()):
            # Remove old packets
            while (self.flow_cache[flow_key] and 
                   self.flow_cache[flow_key][0]['timestamp'] < cutoff_time):
                self.flow_cache[flow_key].popleft()
            
            # Remove empty flows
            if not self.flow_cache[flow_key]:
                del self.flow_cache[flow_key]
        
        # Clean feature cache
        for key in list(self.feature_cache.keys()):
            # Remove old features
            while (self.feature_cache[key] and 
                   self.feature_cache[key][0].get('timestamp', 0) < cutoff_time):
                self.feature_cache[key].popleft()
            
            # Remove empty caches
            if not self.feature_cache[key]:
                del self.feature_cache[key]
