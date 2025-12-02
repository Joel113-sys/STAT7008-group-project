import os
import sys
import platform
from pathlib import Path
import subprocess
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json

class WindowsOCRSetup:
    
    def __init__(self):
        self.system_type = platform.system()
        self.tesseract_path = None
        self.paddle_available = False
        self.tesseract_available = False
        
    def check_system_requirements(self):
        print(f"系统信息: {self.system_type} {platform.release()}")
        print(f"Python版本: {sys.version}")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"系统内存: {memory.total / (1024**3):.1f} GB")
            print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        except:
            print("无法获取内存信息")
    
    def install_tesseract_windows(self):
        tesseract_install_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in tesseract_install_paths:
            if os.path.exists(path):
                self.tesseract_path = path
                print(f"找到Tesseract: {path}")
                return True
        
        print("未找到Tesseract，请按以下步骤安装:")
        print("1. 下载Tesseract安装包: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. 运行安装程序，选择安装中文语言包")
        print("3. 将安装目录添加到系统PATH环境变量")
        return False
    
    def setup_tesseract(self):
        try:
            if self.system_type == "Windows":
                if self.install_tesseract_windows():
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                    self.tesseract_available = True
                    print("Tesseract设置成功")
                    return True
            else:
                import pytesseract
                self.tesseract_available = True
                return True
        except Exception as e:
            print(f"Tesseract设置失败: {e}")
            return False
    
    def setup_paddle_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=True,
                show_log=False
            )
            self.paddle_available = True
            print("PaddleOCR初始化成功")
            return True
        except Exception as e:
            print(f"PaddleOCR初始化失败: {e}")
            print("请安装: pip install paddleocr paddlepaddle")
            return False

class WindowsOCRTools:
    
    def __init__(self):
        self.setup = WindowsOCRSetup()
        self.setup.check_system_requirements()
        self.tesseract_ready = self.setup.setup_tesseract()
        self.paddle_ready = self.setup.setup_paddle_ocr()
        
        self.languages = {
            'english': 'eng',
            'chinese_simple': 'chi_sim', 
            'chinese_traditional': 'chi_tra',
            'japanese': 'jpn',
            'korean': 'kor'
        }
    
    def extract_text_windows(self, image_path: str, language: str = 'eng+chi_sim') -> Dict:
        results = {}
        
        if self.tesseract_ready:
            tesseract_result = self._extract_tesseract_windows(image_path, language)
            results['tesseract'] = tesseract_result
        
        if self.paddle_ready:
            paddle_result = self._extract_paddle_windows(image_path)
            results['paddle'] = paddle_result
        
        best_result = self._select_best_result(results)
        return best_result
    
    def _extract_tesseract_windows(self, image_path: str, language: str) -> Dict:
        try:
            import pytesseract
            
            processed_image = self._preprocess_image_windows(image_path)
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_do_invert=0'
            
            text = pytesseract.image_to_string(
                processed_image, 
                lang=language,
                config=custom_config
            )
            
            detailed_data = pytesseract.image_to_data(
                processed_image,
                lang=language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in detailed_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'success': True,
                'engine': 'tesseract',
                'text': text.strip(),
                'confidence': avg_confidence,
                'language': language
            }
            
        except Exception as e:
            return {
                'success': False,
                'engine': 'tesseract',
                'error': str(e)
            }
    
    def _extract_paddle_windows(self, image_path: str) -> Dict:
        try:
            result = self.setup.paddle_ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return {
                    'success': True,
                    'engine': 'paddle',
                    'text': '',
                    'confidence': 0
                }
            
            all_text = []
            confidences = []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = float(text_info[1])
                        all_text.append(text)
                        confidences.append(confidence)
            
            full_text = '\n'.join(all_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'success': True,
                'engine': 'paddle', 
                'text': full_text,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            return {
                'success': False,
                'engine': 'paddle',
                'error': str(e)
            }
    
    def _preprocess_image_windows(self, image_path: str) -> np.ndarray:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            height, width = image.shape[:2]
            if max(height, width) > 2000:
                scale = 2000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            denoised = cv2.medianBlur(enhanced, 3)
            
            return denoised
            
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def _select_best_result(self, results: Dict) -> Dict:
        successful_results = []
        
        for engine, result in results.items():
            if result['success'] and result.get('text', '').strip():
                successful_results.append(result)
        
        if not successful_results:
            return {
                'success': False,
                'error': '所有OCR引擎都失败了',
                'details': results
            }
        
        successful_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        best_result = successful_results[0]
        best_result['all_results'] = results
        
        return best_result
    
    def create_windows_compatible_visualization(self, image_path: str, ocr_result: Dict) -> str:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            for engine_result in ocr_result.get('all_results', {}).values():
                if engine_result['success'] and 'detailed_data' in engine_result:
                    pass
            
            output_dir = Path("output_files")
            output_dir.mkdir(exist_ok=True)
            
            vis_path = output_dir / f"{Path(image_path).stem}_visualization.png"
            cv2.imwrite(str(vis_path), image)
            
            return str(vis_path)
            
        except Exception as e:
            print(f"创建可视化失败: {e}")
            return ""