import os
import sys
from pathlib import Path
import tempfile
from typing import Dict, List
import PyPDF2
import fitz  
from PIL import Image
import cv2
import numpy as np

from ocr_tools import WindowsOCRTools

class WindowsFileTools:
    
    def __init__(self):
        self.output_dir = "output_files"
        self.temp_dir = tempfile.gettempdir()
        self.ocr_tools = WindowsOCRTools()
        
        Path(self.output_dir).mkdir(exist_ok=True)
        
        print(f"输出目录: {self.output_dir}")
        print(f"临时目录: {self.temp_dir}")
    
    def pdf_extract_text_with_ocr(self, pdf_path: str, use_ocr_for_all: bool = False) -> Dict:
        try:
            if not os.path.exists(pdf_path):
                return {'success': False, 'error': 'PDF文件不存在'}
            
            print(f"处理PDF: {pdf_path}")
            
            direct_text = ""
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        direct_text += f"--- 第 {page_num + 1} 页 ---\n{page_text}\n\n"
            except Exception as e:
                print(f"直接文本提取失败: {e}")
                direct_text = ""
            
            ocr_text = ""
            if use_ocr_for_all or not direct_text.strip():
                print("开始OCR处理...")
                ocr_text = self._ocr_entire_pdf(pdf_path)
            
            final_text = direct_text + ("\n" + "="*50 + "\nOCR结果:\n" + ocr_text if ocr_text else "")
            
            txt_path = Path(pdf_path).with_suffix('').name + '_extracted.txt'
            txt_full_path = Path(self.output_dir) / txt_path
            
            with open(txt_full_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            
            return {
                'success': True,
                'text': final_text,
                'file_path': str(txt_full_path),
                'used_ocr': bool(ocr_text)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PDF处理失败: {str(e)}'}
    
    def _ocr_entire_pdf(self, pdf_path: str) -> str:
        try:
            pdf_document = fitz.open(pdf_path)
            all_ocr_text = ""
            
            for page_num in range(len(pdf_document)):
                print(f"OCR处理第 {page_num + 1}/{len(pdf_document)} 页...")
                
                page = pdf_document[page_num]
                
                mat = fitz.Matrix(2, 2)  
                pix = page.get_pixmap(matrix=mat)
                
                temp_image_path = Path(self.temp_dir) / f"pdf_page_{page_num}.png"
                pix.save(str(temp_image_path))
                
                ocr_result = self.ocr_tools.extract_text_windows(str(temp_image_path))
                
                if ocr_result['success']:
                    page_text = ocr_result['text']
                    all_ocr_text += f"--- 第 {page_num + 1} 页 (OCR) ---\n{page_text}\n\n"
                
                if temp_image_path.exists():
                    temp_image_path.unlink()
            
            pdf_document.close()
            return all_ocr_text
            
        except Exception as e:
            print(f"PDF OCR失败: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str, extract_ocr: bool = True) -> Dict:
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            ocr_results = {}
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                print(f"第 {page_num + 1} 页找到 {len(image_list)} 个图像")
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image_ext = base_image["ext"]
                    image_name = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                    image_path = Path(self.output_dir) / image_name
                    
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    
                    image_paths.append(str(image_path))
                    
                    if extract_ocr:
                        print(f"对图像进行OCR: {image_name}")
                        ocr_result = self.ocr_tools.extract_text_windows(str(image_path))
                        ocr_results[str(image_path)] = ocr_result
                        
                        if ocr_result['success']:
                            txt_path = image_path.with_suffix('.txt')
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(ocr_result['text'])
            
            pdf_document.close()
            
            return {
                'success': True,
                'image_paths': image_paths,
                'ocr_results': ocr_results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def process_screenshot_ocr(self, screenshot_path: str) -> Dict:
        try:
            if not os.path.exists(screenshot_path):
                return {'success': False, 'error': '截图文件不存在'}
            
            print(f"处理截图: {screenshot_path}")
            
            image = cv2.imread(screenshot_path)
            if image is None:
                return {'success': False, 'error': '无法读取截图'}

            height, width = image.shape[:2]
            if max(height, width) > 1920:
                scale = 1920 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                optimized_path = Path(self.temp_dir) / "optimized_screenshot.png"
                cv2.imwrite(str(optimized_path), image)
                screenshot_path = str(optimized_path)
            
            ocr_result = self.ocr_tools.extract_text_windows(screenshot_path)
            
            if ocr_result['success']:
                result_path = Path(screenshot_path).with_suffix('').name + '_ocr.txt'
                result_full_path = Path(self.output_dir) / result_path
                
                with open(result_full_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_result['text'])
                
                ocr_result['result_file'] = str(result_full_path)
            
            return ocr_result
            
        except Exception as e:
            return {'success': False, 'error': f'截图OCR失败: {str(e)}'}

    def download_pdf(self, url: str, filename: str = None) -> Dict:
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if filename is None:
                filename = url.split('/')[-1] or "downloaded_file.pdf"
            if not filename.endswith('.pdf'):
                filename += '.pdf'
                
            pdf_path = Path(self.output_dir) / filename
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF下载成功: {pdf_path}")
            return {
                'success': True,
                'file_path': str(pdf_path),
                'filename': filename
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PDF下载失败: {str(e)}'}

    def save_image(self, image_data, filename: str) -> Dict:
        try:
            image_path = Path(self.output_dir) / filename
            
            if isinstance(image_data, np.ndarray):
                cv2.imwrite(str(image_path), image_data)
            elif isinstance(image_data, Image.Image):
                image_data.save(str(image_path))
            else:
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            
            print(f"图像保存成功: {image_path}")
            return {
                'success': True,
                'file_path': str(image_path)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'图像保存失败: {str(e)}'}

    def write_text_file(self, filename: str, content: str) -> Dict:
        try:
            file_path = Path(self.output_dir) / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"文本文件保存成功: {file_path}")
            return {
                'success': True,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'文本文件保存失败: {str(e)}'}

    def pdf_extract_text(self, pdf_path: str) -> Dict:
        try:
            if not os.path.exists(pdf_path):
                return {'success': False, 'error': 'PDF文件不存在'}
            
            print(f"提取PDF文本: {pdf_path}")
            
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"--- 第 {page_num + 1} 页 ---\n{page_text}\n\n"
            
            txt_path = Path(pdf_path).with_suffix('').name + '_text_only.txt'
            txt_full_path = Path(self.output_dir) / txt_path
            
            with open(txt_full_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            return {
                'success': True,
                'text': text,
                'file_path': str(txt_full_path)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'PDF文本提取失败: {str(e)}'}