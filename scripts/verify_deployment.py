#!/usr/bin/env python3
"""
Deployment Verification Script for PM2.5 MLP Dashboard
=====================================================

Este script verifica que la aplicación esté correctamente desplegada
y funcional en Render o cualquier otro entorno de producción.
"""

import requests
import json
import time
import sys
import argparse
from typing import Dict, List, Tuple, Any


class DeploymentVerifier:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results = []
        
    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp and level"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        colors = {
            'INFO': '\033[0;34m',    # Blue
            'SUCCESS': '\033[0;32m', # Green
            'WARNING': '\033[1;33m', # Yellow
            'ERROR': '\033[0;31m',   # Red
            'NC': '\033[0m'          # No Color
        }
        
        color = colors.get(level, colors['NC'])
        print(f"{color}[{timestamp}] {level}: {message}{colors['NC']}")
        
    def add_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Add test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': time.time()
        }
        self.results.append(result)
        level = 'SUCCESS' if success else 'ERROR'
        self.log(f"{test_name}: {'✅ PASSED' if success else '❌ FAILED'} - {message}", level)
        
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.add_result(
                        'Health Check', 
                        True, 
                        'Endpoint responded correctly',
                        {'status': data.get('status'), 'version': data.get('version')}
                    )
                    return True
                else:
                    self.add_result('Health Check', False, f"Unhealthy status: {data.get('status')}")
                    return False
            else:
                self.add_result('Health Check', False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.add_result('Health Check', False, str(e))
            return False
            
    def test_main_page(self) -> bool:
        """Test main page loads"""
        try:
            response = requests.get(self.base_url, timeout=self.timeout)
            if response.status_code == 200:
                content = response.text.lower()
                if 'sistema de entrenamiento' in content or 'pm2.5' in content:
                    self.add_result('Main Page', True, 'Page loaded and contains expected content')
                    return True
                else:
                    self.add_result('Main Page', False, 'Page loaded but content not as expected')
                    return False
            else:
                self.add_result('Main Page', False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.add_result('Main Page', False, str(e))
            return False
            
    def test_static_files(self) -> bool:
        """Test static files are accessible"""
        static_files = [
            '/static/css/styles.css',
            '/static/js/script.js'
        ]
        
        all_passed = True
        for file_path in static_files:
            try:
                response = requests.get(f"{self.base_url}{file_path}", timeout=self.timeout)
                if response.status_code == 200:
                    self.add_result(f"Static File: {file_path}", True, 'File accessible')
                else:
                    self.add_result(f"Static File: {file_path}", False, f"HTTP {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.add_result(f"Static File: {file_path}", False, str(e))
                all_passed = False
                
        return all_passed
        
    def test_api_endpoints(self) -> bool:
        """Test API endpoints are accessible"""
        endpoints = [
            ('/upload', 'POST'),
            ('/train', 'POST'),
            ('/deploy', 'POST'),
            ('/backtest', 'POST'),
            ('/download', 'GET')
        ]
        
        all_passed = True
        for endpoint, method in endpoints:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=self.timeout)
                    
                # Expect 400 or 405 (method not allowed) or 200, but not 404
                if response.status_code in [200, 400, 405]:
                    self.add_result(f"API Endpoint: {endpoint}", True, f"HTTP {response.status_code}")
                else:
                    self.add_result(f"API Endpoint: {endpoint}", False, f"HTTP {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.add_result(f"API Endpoint: {endpoint}", False, str(e))
                all_passed = False
                
        return all_passed
        
    def test_ssl_certificate(self) -> bool:
        """Test SSL certificate validity"""
        try:
            import ssl
            import socket
            from datetime import datetime
            
            hostname = self.base_url.replace('https://', '').replace('http://', '').split('/')[0]
            
            if not self.base_url.startswith('https://'):
                self.add_result('SSL Certificate', True, 'HTTP detected (not recommended for production)')
                return True
                
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Check expiration
                    expire_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expire = (expire_date - datetime.utcnow()).days
                    
                    if days_until_expire > 30:
                        self.add_result('SSL Certificate', True, f'Valid, expires in {days_until_expire} days')
                        return True
                    else:
                        self.add_result('SSL Certificate', False, f'Expires in {days_until_expire} days')
                        return False
        except Exception as e:
            self.add_result('SSL Certificate', False, str(e))
            return False
            
    def test_response_time(self) -> bool:
        """Test response time is acceptable"""
        try:
            start_time = time.time()
            response = requests.get(self.base_url, timeout=self.timeout)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # ms
            
            if response.status_code == 200 and response_time < 3000:  # 3 seconds
                self.add_result('Response Time', True, f'{response_time:.0f}ms')
                return True
            else:
                self.add_result('Response Time', False, f'{response_time:.0f}ms (too slow)')
                return False
        except Exception as e:
            self.add_result('Response Time', False, str(e))
            return False
            
    def test_cors_headers(self) -> bool:
        """Test CORS headers are properly set"""
        try:
            response = requests.options(self.base_url, timeout=self.timeout)
            
            # Check for CORS headers
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            missing_headers = []
            for header in cors_headers:
                if header not in response.headers:
                    missing_headers.append(header)
                    
            if not missing_headers:
                self.add_result('CORS Headers', True, 'All required headers present')
                return True
            else:
                self.add_result('CORS Headers', False, f'Missing: {", ".join(missing_headers)}')
                return False
        except Exception as e:
            self.add_result('CORS Headers', False, str(e))
            return False
            
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all verification tests"""
        self.log("🚀 Starting deployment verification...", 'INFO')
        self.log(f"Testing URL: {self.base_url}", 'INFO')
        print()
        
        tests = [
            ('Health Check', self.test_health_check),
            ('Main Page', self.test_main_page),
            ('Static Files', self.test_static_files),
            ('API Endpoints', self.test_api_endpoints),
            ('SSL Certificate', self.test_ssl_certificate),
            ('Response Time', self.test_response_time),
            ('CORS Headers', self.test_cors_headers)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...", 'INFO')
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.add_result(test_name, False, f"Test execution failed: {str(e)}")
            print()
            
        # Generate summary
        success_rate = (passed / total) * 100
        
        summary = {
            'url': self.base_url,
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': success_rate,
            'overall_status': 'HEALTHY' if success_rate >= 80 else 'WARNING' if success_rate >= 60 else 'CRITICAL',
            'results': self.results,
            'timestamp': time.time()
        }
        
        self.print_summary(summary)
        return summary
        
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        print("=" * 60)
        print("📊 DEPLOYMENT VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"URL: {summary['url']}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Status: {summary['overall_status']}")
        print()
        
        # Detailed results
        print("📋 DETAILED RESULTS:")
        print("-" * 60)
        for result in summary['results']:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"{status} - {result['test']}")
            print(f"    {result['message']}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 60)
        if summary['success_rate'] < 100:
            print("• Fix failed tests before deploying to production")
            print("• Ensure all static files are properly served")
            print("• Verify API endpoints handle errors gracefully")
        if summary['success_rate'] < 80:
            print("• Review SSL certificate configuration")
            print("• Optimize response times")
        if summary['success_rate'] < 60:
            print("• ⚠️  Critical issues detected - do not deploy!")
            print("• Review application logs for errors")
            print("• Check server resources and configuration")
        print()
        
        # Save results
        with open('deployment_verification_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.log("Results saved to deployment_verification_results.json", 'SUCCESS')


def main():
    parser = argparse.ArgumentParser(description='Verify PM2.5 MLP Dashboard deployment')
    parser.add_argument('--url', required=True, help='Base URL of the deployed application')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout for requests (default: 30s)')
    
    args = parser.parse_args()
    
    verifier = DeploymentVerifier(args.url, args.timeout)
    summary = verifier.run_all_tests()
    
    # Exit with appropriate code
    if summary['overall_status'] == 'HEALTHY':
        sys.exit(0)
    elif summary['overall_status'] == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()