#!/usr/bin/env python3
"""
데이터 시각화 및 증강 도구 실행 스크립트
백엔드와 프론트엔드를 함께 실행합니다.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def run_backend():
    """백엔드 서버를 실행합니다."""
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    print("🚀 백엔드 서버를 시작합니다...")
    print(f"📁 백엔드 디렉토리: {backend_dir}")
    
    try:
        # 의존성 설치 확인
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        # 백엔드 서버 실행
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 백엔드 실행 중 오류 발생: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 백엔드 서버가 중단되었습니다.")
        return False

def run_frontend():
    """프론트엔드 애플리케이션을 실행합니다."""
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    print("🚀 프론트엔드 애플리케이션을 시작합니다...")
    print(f"📁 프론트엔드 디렉토리: {frontend_dir}")
    
    try:
        # 의존성 설치 확인
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        # Streamlit 애플리케이션 실행
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", 
                       "--server.port", "8501", "--server.address", "localhost"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 프론트엔드 실행 중 오류 발생: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 프론트엔드 애플리케이션이 중단되었습니다.")
        return False

def check_dependencies():
    """필요한 의존성이 설치되어 있는지 확인합니다."""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "pandas", 
        "numpy", "plotly", "requests", "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}")
        print("📦 패키지를 설치하려면: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 모든 의존성이 설치되어 있습니다.")
    return True

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("📊 데이터 시각화 및 증강 도구")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🔧 실행 모드를 선택하세요:")
    print("1. 백엔드만 실행 (FastAPI 서버)")
    print("2. 프론트엔드만 실행 (Streamlit 앱)")
    print("3. 백엔드 + 프론트엔드 동시 실행")
    print("4. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-4): ").strip()
            
            if choice == "1":
                print("\n🔧 백엔드만 실행합니다...")
                run_backend()
                break
                
            elif choice == "2":
                print("\n🔧 프론트엔드만 실행합니다...")
                print("⚠️  백엔드 서버가 실행되지 않은 경우 일부 기능이 작동하지 않을 수 있습니다.")
                run_frontend()
                break
                
            elif choice == "3":
                print("\n🔧 백엔드와 프론트엔드를 동시에 실행합니다...")
                
                # 백엔드를 별도 스레드에서 실행
                backend_thread = threading.Thread(target=run_backend, daemon=True)
                backend_thread.start()
                
                # 백엔드 시작 대기
                print("⏳ 백엔드 서버 시작을 기다리는 중...")
                time.sleep(3)
                
                # 프론트엔드 실행
                run_frontend()
                break
                
            elif choice == "4":
                print("👋 프로그램을 종료합니다.")
                sys.exit(0)
                
            else:
                print("❌ 잘못된 선택입니다. 1-4 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            sys.exit(0)

if __name__ == "__main__":
    main() 