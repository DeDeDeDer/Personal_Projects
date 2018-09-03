"""Scans local Wifi network"""
import nmap

"""Assumptions"""
# 1. Assumes you know local network subnet mask
# 2. Code runs via Root/user for Kali Linux OR sudo for MacOS

"""How to run code"""
# Using cmd terminal type either:
# sudo python3 Wifi_Dev_Scan.py - requires manual local user PW user input


"""Pending Modifications"""
# 1. Add subprocess.check_output() module to run code via python shell


"""Operation"""
nm = nmap.PortScanner()
cidr2 = '192.168.1.1/24'
a = nm.scan(hosts=cidr2, arguments='-sP')   # 1-1024

for k, v in a['scan'].items():
    if str(v['status']['state']) == 'up':
        # print(str(v))
        try:
            print(str(v['hostnames'][0]['name']) + ' => ' +
                  str(v['vendor'][str(v['addresses']['mac'])]) + ' => ' +
                  str(v['addresses']['ipv4']) + ' => ' +
                  str(v['addresses']['mac']))
        except:
            print('Device Name N/A' + ' => ' +
                  'MAC Add N/A' + ' => ' +
                  str(v['addresses']['ipv4']) + ' => ' +
                  'MAC Add N/A')
