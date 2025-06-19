# main.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QMimeData
import preprocessor
import ocr
import cv2
import numpy as np


class DropLabel(QLabel):
    """支持拖拽的标签控件"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                font-size: 16px;
                color: #777;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #f8f8f8;
            }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4CAF50;
                    border-radius: 10px;
                    padding: 20px;
                    font-size: 16px;
                    background-color: #f0fff0;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                font-size: 16px;
                color: #777;
            }
        """)

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        main_window = self.window()  # 获取顶层窗口对象
        if hasattr(main_window, 'load_image'):
            main_window.load_image(file_path)
        else:
            print("Error: main window has no method 'load_image'")


class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR 文本识别系统")
        self.setGeometry(100, 100, 1400, 900)

        # 主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("OCR 文本识别系统")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                text-align: center;
            }
        """)
        main_layout.addWidget(title_label)

        # 操作按钮布局
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("选择图像")
        self.process_btn = QPushButton("预处理图像")
        self.ocr_btn = QPushButton("运行OCR")
        self.save_btn = QPushButton("保存结果")
        self.clear_btn = QPushButton("清空")

        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """
        self.load_btn.setStyleSheet(button_style)
        self.process_btn.setStyleSheet(button_style)
        self.ocr_btn.setStyleSheet(button_style)
        self.save_btn.setStyleSheet(button_style)
        self.clear_btn.setStyleSheet(button_style)

        # 按钮连接事件
        self.load_btn.clicked.connect(self.select_image)
        self.process_btn.clicked.connect(self.preprocess_image)
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.save_btn.clicked.connect(self.save_results)
        self.clear_btn.clicked.connect(self.clear_all)

        # 添加按钮
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.ocr_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        main_layout.addLayout(button_layout)

        # 拖拽提示
        drag_label = QLabel("或者拖放图像文件到下方区域")
        drag_label.setAlignment(Qt.AlignCenter)
        drag_label.setStyleSheet("font-size: 14px; color: #7f8c8d; padding: 5px;")
        main_layout.addWidget(drag_label)

        # 图像显示区域
        image_layout = QHBoxLayout()
        self.original_frame = QFrame()
        self.original_frame.setLayout(QVBoxLayout())
        self.original_frame.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # 原始图像区域
        orig_title = QLabel("原始图像")
        orig_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        orig_title.setAlignment(Qt.AlignCenter)
        self.original_frame.layout().addWidget(orig_title)

        self.original_label = DropLabel("拖放图像文件到这里\n或点击上方按钮选择")
        self.original_frame.layout().addWidget(self.original_label)

        # 处理后图像区域
        self.processed_frame = QFrame()
        self.processed_frame.setLayout(QVBoxLayout())
        self.processed_frame.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        proc_title = QLabel("处理后图像")
        proc_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        proc_title.setAlignment(Qt.AlignCenter)
        self.processed_frame.layout().addWidget(proc_title)

        self.processed_label = QLabel("预处理后图像将显示在这里")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        self.processed_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                background-color: #fff;
                color: #95a5a6;
                font-size: 14px;
                padding: 20px;
            }
        """)
        self.processed_frame.layout().addWidget(self.processed_label)

        image_layout.addWidget(self.original_frame)
        image_layout.addWidget(self.processed_frame)
        main_layout.addLayout(image_layout)

        # OCR结果区域
        result_frame = QFrame()
        result_frame.setLayout(QVBoxLayout())
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
            }
        """)

        result_title = QLabel("OCR识别结果")
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        result_title.setAlignment(Qt.AlignCenter)
        result_frame.layout().addWidget(result_title)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("识别结果将显示在这里...")
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
                min-height: 200px;
            }
        """)
        result_frame.layout().addWidget(self.result_text)
        main_layout.addWidget(result_frame)

        # 状态变量
        self.original_image = None
        self.processed_image = None
        self.ocr_results = []
        self.current_image_path = None

        # 禁用初始按钮
        self.process_btn.setEnabled(False)
        self.ocr_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def select_image(self):
        """通过文件资源管理器选择图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """加载图像文件"""
        try:
            # 使用OpenCV读取图像
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)

            if self.original_image is None:
                 # 尝试使用 fromfile + imdecode 兼容中文路径
                abs_path = os.path.abspath(file_path)
                data = np.fromfile(abs_path, dtype=np.uint8)
                self.original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

                if self.original_image is None:
                    raise ValueError(f"无法读取图像文件: {file_path}")

            # 显示原始图像
            self.display_image(self.original_image, self.original_label)

            # 更新标签文本
            filename = os.path.basename(file_path)
            self.original_label.setText(f"已加载: {filename}")
            self.original_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3498db;
                    border-radius: 10px;
                    background-color: #f0f8ff;
                    font-size: 14px;
                    color: #2c3e50;
                }
            """)

            # 重置处理结果
            self.processed_image = None
            self.processed_label.setText("预处理后图像将显示在这里")
            self.processed_label.setStyleSheet("""
                QLabel {
                    border: 1px solid #ddd;
                    background-color: #fff;
                    color: #95a5a6;
                    font-size: 14px;
                    padding: 20px;
                }
            """)
            self.result_text.clear()
            self.ocr_results = []

            # 启用按钮
            self.process_btn.setEnabled(True)
            self.ocr_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            self.original_label.setText("拖放图像文件到这里\n或点击上方按钮选择")
            self.original_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 10px;
                    padding: 20px;
                    font-size: 16px;
                    color: #777;
                }
            """)

    def preprocess_image(self):
        if self.original_image is None:
            return

        try:
            # 使用新函数处理内存中的图像
            self.processed_image = preprocessor.preprocess_image_from_array(self.original_image)

            # 显示处理后的图像
            self.display_image(self.processed_image, self.processed_label)

            # 更新标签
            self.processed_label.setText("")
            self.processed_label.setStyleSheet("")

            # 启用OCR按钮
            self.ocr_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像预处理失败: {str(e)}")

    def run_ocr(self):
        if self.processed_image is None:
            return

        try:
            # 运行OCR
            self.ocr_results = ocr.run_ocr(self.processed_image)

            # 显示结果
            result_text = "\n\n".join(self.ocr_results)
            self.result_text.setPlainText(result_text)

            # 启用保存按钮
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"OCR识别失败: {str(e)}")

    def save_results(self):
        if not self.ocr_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存OCR结果", "",
            "文本文件 (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(self.ocr_results))
                QMessageBox.information(self, "成功", f"结果已成功保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def clear_all(self):
        """清空所有内容"""
        self.original_image = None
        self.processed_image = None
        self.ocr_results = []
        self.current_image_path = None

        self.original_label.setText("拖放图像文件到这里\n或点击上方按钮选择")
        self.original_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                font-size: 16px;
                color: #777;
            }
        """)

        self.processed_label.setText("预处理后图像将显示在这里")
        self.processed_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                background-color: #fff;
                color: #95a5a6;
                font-size: 14px;
                padding: 20px;
            }
        """)

        self.result_text.clear()

        self.process_btn.setEnabled(False)
        self.ocr_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def display_image(self, image, label):
        """在QLabel中显示OpenCV图像"""
        # 转换图像格式
        if len(image.shape) == 2:  # 灰度图
            h, w = image.shape
            bytes_per_line = w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图
            # 将BGR转换为RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 创建并缩放Pixmap
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 设置Pixmap
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ecf0f1;
        }
        QGroupBox {
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
    """)

    window = OCRApp()
    window.show()
    sys.exit(app.exec_())