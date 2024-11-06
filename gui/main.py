import remi.gui as gui
from remi import start, App
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

class BitTunerApp(App):
    def __init__(self, *args):
        self.plot_key_index = gui.Image('my_res:sift_1M_4_8_1.svg', width='95%', height='220px',
                                   style={'margin-top': '10px', 'margin-left': '10px'})
        self.plot_gap_distribution = gui.Image('my_res:sift_1M_4_8_2.svg', width='95%', height='220px',
                                          style={'margin-top': '10px', 'margin-left': '10px'})
        self.filename = 'none'
        self.partition_table = gui.Table.new_from_list([["Partition ID", "Mean", "Variance"]],
            style = {'width': 'calc(100% - 10px)', 'margin-top': '10px', 'margin': '5px 10px 5px 5px', 'background-color': '#e0f7fa' },
            header_style={'background-color': '#ffffff'} )
        self.partition_visual = gui.Image(
            'my_res:output_image_gaps_matplotlib_with_pelt2_1M_3.svg',
            style={
                'width': 'calc(100% - 5px)', 'height': '300px', 'margin': '5px 2px',
                'background-color': 'transparent', 'color': '#4e5258', 'font-family': 'Helvetica',
                'padding': '5px', 'font-size': '16px', 'font-weight': 'bold'
            }
        )
        self.sample_ratio_input = gui.TextInput(style={
            'width': '95%', 'height': '30px', 'margin': '5px 10px',
            'background-color': 'transparent', 'border': '1px solid #e0e0e0',
            'color': '#4e5258', 'font-family': 'Helvetica', 'padding': '5px',
            'font-size': '16px', 'line-height': '30px'
        })
        self.allocation_table = gui.Table.new_from_list(
            [["Partition ID", 'Starting Index', 'Ending Index', "Residual Bits"],[" TBD", "TBD", 'TBD', 'TBD'],
             [" TBD", "TBD", 'TBD', 'TBD'],[" TBD", "TBD", 'TBD', 'TBD'],[" TBD", "TBD", 'TBD', 'TBD'],[" TBD", "TBD", 'TBD', 'TBD'],],
            style={'width': 'calc(100% - 10px)', 'margin-top': '10px', 'margin': '5px 10px 5px 5px',
                   'background-color': '#e0f7fa'},
            header_style={'background-color': '#ffffff'}
        )
        self.benchmark_result = gui.Table.new_from_list(
            [["Algorithm", "Compression Ratio", "Compression Time", "Decompression Time"], ["TBD", 'TBD','TBD', 'TBD'],
             # 空白行
             [" TBD", "TBD", 'TBD', 'TBD'],
             [" TBD", "TBD", 'TBD', 'TBD']],
            style={'width': 'calc(100% - 10px)', 'margin-top': '10px', 'height': '30px','margin-left': '10px', 'margin-right': '10px'})

        self.allocation_chart = gui.Image('my_res:1_CR-RB.svg', style={
            'width': 'auto',
            'height': '150px',
            'display': 'block',
            'margin': '10px auto',
            'background-color': 'transparent',
            'color': '#4e5258',
            'font-family': 'Helvetica',
            'font-size': '16px',
            'font-weight': 'bold'
        })

        checkbox_style = {
            'padding-left': '15px',
            'font-size': '16px',
            'font-family': 'Helvetica',
            'color': '#4e5258',
        }

        self.lz4_checkbox = gui.CheckBoxLabel("LZ4", False, width='auto', height='20px', margin='5px',
                                              style=checkbox_style)
        self.lzma_checkbox = gui.CheckBoxLabel("LZMA", False, width='auto', height='20px', margin='5px',
                                               style=checkbox_style)
        self.Huffman_checkbox = gui.CheckBoxLabel("Huffman", False, width='auto', height='20px', margin='5px',
                                                  style=checkbox_style)
        self.Delta_checkbox = gui.CheckBoxLabel("Delta", False, width='auto', height='20px', margin='5px',
                                                  style=checkbox_style)
        super(BitTunerApp, self).__init__(*args, static_file_path= {'my_res': './res/'})

    def main(self):
        main_container = gui.Container(
            width='100%',
            height='100%',
            margin='auto',
            style={
                'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
                'margin-top': '0px',
                'overflow': 'hidden'
            }
        )

        grid_container = gui.Container(
            width='100%',
            height='100%',
            style={
                'display': 'grid',
                'grid-template-rows': '70px 1fr',
                'gap': '0px',
                'background-color': '#ededed'
            }
        )

        title = gui.Label(
            "BitTuner",
            style={
                'font-size': '30px',
                'font-weight': '1000',
                'font-family': 'Helvetica Neue',
                'text-align': 'left',
                'color': '#ffffff',
                'background-color': '#3E5992',
                'padding': '15px',
                'display': 'flex',
                'align-items': 'center',
            }
        )

        grid_container.append(title)

        container = gui.Container(
            style={
                'display': 'grid',
                'margin-top': '15px',
                'margin-bottom': '30px',
                'grid-template-columns': '1fr 1fr 1fr',
                'gap': '8px',
                'padding': '0 5px',
                'background-color': '#ededed'
            }
        )

        grid_container.append(container)

        #数据探索部分
        explore_container = self.create_section("Data Exploration")
        horizontal_line = gui.Container(width='95%', height='2px',
                                        style={
                                            'background-color': '#d3d3d3',  # 横线颜色
                                            'margin-top': '10px',  # 距离标题下方10像素
                                            'margin-left': 'auto',  # 水平居中
                                            'margin-right': 'auto'
                                        })# 添加横线
        explore_container.append(horizontal_line)  # 将横线添加到数据探索容器
        dataset_selector = gui.DropDown.new_from_list(
            ('SIFT1B', 'Deep1B', 'other datasets'),
            style={
                'width': '95%',  # 宽度设置为 80%
                'height': '40px',  # 高度设置为 30px
                'margin': '5px 10px',  # 设置上下左右的外边距
                'background-color': 'transparent',  # 透明背景
                'border': '1px solid #e0e0e0',  # 单层边框，颜色为深灰色
                'color': '#d3d3d3',  # 字体颜色
                'font-family': 'Helvetica',  # 字体设置为 Helvetica
                'padding': '5px',  # 内边距以保持内容和边框的距离
                'font-size': '16px',  # 字体大小
                'font-weight': 'bold'  # 字体粗细
            }
        )
        dataset_selector.onchange.do(self.on_dataset_selected)
        explore_container.append(gui.Label("Pre-load Dataset", style={'font-weight': '900', 'font-family': 'Helvetica Neue', 'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px', 'margin-left': '10px'}))
        explore_container.append(dataset_selector)

        upload_button = gui.FileUploader('./file/', label='Upload File',width=200, height=30, margin='10px')
        upload_button.onsuccess.do(self.fileupload_on_success)
        upload_button.onfailed.do(self.fileupload_on_failed)
        upload_button.attributes['value'] = "Upload File"
        explore_container.append(gui.Label("Or Custom Dataset", style={'font-weight': '900', 'font-family': 'Helvetica Neue', 'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px', 'margin-left': '10px'}))
        explore_container.append(upload_button)

        confirm_button = gui.Button("Submit", width='40%', height='40px', color ='#75b9f3', margin = 'auto',
                                    style={
                                        'background-color': '#75b9f3',
                                        'color': '#ffffff',
                                        'margin-top': '10px',
                                        'margin-left': 'auto',
                                        'margin-right': 'auto',
                                        'display': 'block',
                                        'text-align': 'center',
                                        'font-family': 'Helvetica',
                                        'font-size': '18px',
                                        'font-weight': 'bold',
                                        'color': '#f0f0f0',
                                        'box-shadow': 'none'
                                    }
                                    )
        confirm_button.onclick.do(self.on_confirm_button_click)
        explore_container.append(confirm_button)
        confirm_button.attributes['onmouseover'] = "this.style.backgroundColor='#3a66a7';"
        confirm_button.attributes['onmouseout'] = "this.style.backgroundColor='#75b9f3';"

        # plot_key_index = gui.Image('/res:key_index_plot.png', width='100%', height='120px', style={'margin-top': '10px'})
        # plot_gap_distribution = gui.Image('/res:gap_distribution_plot.png', width='100%', height='120px', style={'margin-top': '10px'})
        explore_container.append(gui.Label("Visualization",
                                           style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                  'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                  'margin-left': '10px'}))
        explore_container.append(self.plot_key_index)
        explore_container.append(self.plot_gap_distribution)

        container.append(explore_container)


        #键分区部分
        partition_container = self.create_section("Partition")
        horizontal_line_partition = gui.Container(width='95%', height='2px',
                                        style={
                                            'background-color': '#d3d3d3',
                                            'margin-top': '10px',
                                            'margin-left': '10px',
                                            'margin-right': '10px'
                                        })  # 添加横线
        partition_container.append(horizontal_line_partition)
        partition_container.append(gui.Label("Sampling Ratio(%):",
                                           style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                  'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                  'margin-left': '10px'}))

        partition_container.append(self.sample_ratio_input)
        partition_button = gui.Button("Partition", width='50%', height='40px', color='#75b9f3', margin='auto',
                                    style={
                                        'background-color': '#75b9f3',
                                        'color': '#ffffff',
                                        'margin-top': '20px',
                                        'margin-left': 'auto',
                                        'margin-right': 'auto',
                                        'display': 'block',
                                        'text-align': 'center',
                                        'font-family': 'Helvetica',
                                        'font-size': '18px',
                                        'font-weight': 'bold',
                                        'color': '#f0f0f0',
                                        'box-shadow': 'none'
                                    }
                                    )
        partition_button.attributes['onmouseover'] = "this.style.backgroundColor='#3a66a7';"
        partition_button.attributes['onmouseout'] = "this.style.backgroundColor='#75b9f3';"
        partition_button.onclick.do(self.on_partition_button_click)
        partition_container.append(partition_button)


        partition_container.append(gui.Label("Partition Result", style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                  'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                  'margin-left': '10px'}))
        partition_container.append(self.partition_visual)

        # 确保文件路径正确
        stats_file_path = "./res/segment_stats.txt"

        try:
            with open(stats_file_path, 'r') as f:

                header = next(f).strip()
                print("Header:", header)

                lines = f.readlines()
                row_count = len(lines)
                row_height = max(20, 200 // row_count)

                partition_id = 1
                for line in lines:
                    # 解析数据行
                    print("Raw line:", line)
                    partition_idx,start_idx, end_idx,mean, std_dev = line.strip().split('\t')
                    print(f"Parsed: Partiton id={partition_idx}, Start Index={start_idx}, End Index={end_idx}, Mean={mean}, Std Dev={std_dev}")

                    # 创建新的表格行并添加到表格中
                    row = gui.TableRow(style={'height': f'{row_height}px'})
                    row.append(gui.TableItem(f"P{partition_id}"))
                    row.append(gui.TableItem(mean))
                    row.append(gui.TableItem(std_dev))
                    self.partition_table.append(row)
                    partition_id += 1
        except FileNotFoundError:
            print("统计信息文件未找到。")
        except Exception as e:
            print(f"Error reading file: {e}")


        partition_container.append(self.partition_table)


        container.append(partition_container)


        #残差位优化部分
        benchmark_container = self.create_section("Allocation and Benchmark")
        horizontal_line_benchmark = gui.Container(width='95%', height='2px',
                                        style={
                                            'background-color': '#d3d3d3',
                                            'margin-top': '10px',
                                            'margin-left': '10px',
                                            'margin-right': '10px'
                                        })  # 添加横线
        benchmark_container.append(horizontal_line_benchmark)

        allocation_button = gui.Button("Optimize", width='40%', height='40px', color='#75b9f3', margin='auto',
                                    style={
                                        'background-color': '#75b9f3',
                                        'color': '#ffffff',
                                        'margin-top': '30px',
                                        'margin-left': 'auto',
                                        'margin-right': 'auto',
                                        'display': 'block',
                                        'text-align': 'center',
                                        'font-family': 'Helvetica',
                                        'font-size': '18px',
                                        'font-weight': 'bold',
                                        'color': '#f0f0f0',
                                        'box-shadow': 'none'
                                    }
                                    )

        allocation_button.attributes['onmouseover'] = "this.style.backgroundColor='#3a66a7';"
        allocation_button.attributes['onmouseout'] = "this.style.backgroundColor='#75b9f3';"
        allocation_button.onclick.do(self.on_allocation_button_click)

        benchmark_container.append(allocation_button)

        benchmark_container.append(
            gui.Label("Allocation Result", style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                  'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                  'margin-left': '10px'}))


        stats_file_path = "./res/segment_stats.txt"

        try:
            with open(stats_file_path, 'r') as f:

                header = next(f).strip()
                print("Header:", header)


                lines = f.readlines()
                row_count = len(lines)
                row_height = max(20, 200 // row_count)
                partition_id = 1
                for line in lines:

                    print("Raw line:", line)
                    start_idx, mean, std_dev = line.strip().split('\t')
                    print(f"Parsed: Start Index={start_idx}, Mean={mean}, Std Dev={std_dev}")


                    row = gui.TableRow(style={'height': f'{row_height}px'})
                    row.append(gui.TableItem(f"P{partition_id}"))
                    row.append(gui.TableItem(mean))
                    row.append(gui.TableItem(std_dev))
                    self.allocation_table.append(row)
                    partition_id += 1
        except FileNotFoundError:
            print("统计信息文件未找到。")
        except Exception as e:
            print(f"Error reading file: {e}")

        benchmark_container.append(self.allocation_table)


        benchmark_container.append(self.allocation_chart)

        checkbox_container = gui.Container(style={
            'display': 'flex',
            'flex-direction': 'row',
            'justify-content': 'center',
            'align-items': 'center',
            'gap': '10px'
        })

        checkbox_container.append(self.lz4_checkbox)
        checkbox_container.append(self.lzma_checkbox)
        checkbox_container.append(self.Delta_checkbox)
        checkbox_container.append(self.Huffman_checkbox)
        benchmark_container.append(
            gui.Label("Choose Baseline：", style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                 'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                 'margin-left': '10px'}))
        benchmark_container.append(checkbox_container)

        benchmark_button = gui.Button("Benchmark", width='40%', height='40px', color='#75b9f3', margin='auto',
                                      style={
                                          'background-color': '#75b9f3',
                                          'color': '#ffffff',
                                          'margin-top': '20px',
                                          'margin-left': 'auto',
                                          'margin-right': 'auto',
                                          'display': 'block',
                                          'text-align': 'center',
                                          'font-family': 'Helvetica',
                                          'font-size': '18px',
                                          'font-weight': 'bold',
                                          'color': '#f0f0f0',
                                          'box-shadow': 'none'
                                      }
                                      )

        benchmark_button.attributes['onmouseover'] = "this.style.backgroundColor='#3a66a7';"
        benchmark_button.attributes['onmouseout'] = "this.style.backgroundColor='#75b9f3';"

        benchmark_button.onclick.do(self.on_benchmark_button_click)
        benchmark_container.append(benchmark_button)

        benchmark_container.append(gui.Label("Benchmark Result：", style={'font-weight': '900', 'font-family': 'Helvetica Neue',
                                                  'font-size': '20px', 'color': '#4e5258', 'margin-top': '25px',
                                                  'margin-left': '10px'}))
        benchmark_container.append(self.benchmark_result)

        container.append(benchmark_container)

        main_container.append(grid_container)
        return main_container


    def create_section(self, title_text):
        section_style = {'margin': '10px','margin-top':'5px', 'padding': '10px', 'background-color': '#fff',
                         'border-radius': '5px', 'box-shadow': '0 2px 5px rgba(0,0,0,0.2)'}
        section = gui.Container(style=section_style)
        section.append(gui.Label(title_text, style={'font-weight': '900', 'font-family': 'Helvetica Neue', 'font-size': '30px', 'color': '#4e5258', 'margin-top': '15px', 'margin-left': '10px'}))
        return section

    def on_dataset_selected(self, dropdown, value):
        """ 当用户从下拉列表选择一个数据集时触发 """
        self.selected_dataset = dropdown.get_value()
        print("Selected dataset:", self.selected_dataset)

    def on_confirm_button_click(self, button):
        """确认所选数据集并加载相应图像"""
        if getattr(self, 'selected_dataset', '') == 'SIFT1B':
            self.plot_key_index.attributes['src'] = 'my_res:sift_1B_4_8_1.svg'
            self.plot_gap_distribution.attributes['src'] = 'my_res:sift_1B_4_8_2.svg'
            print("Images for SIFT1B dataset loaded.")
        elif getattr(self, 'selected_dataset', '') == 'Deep1B':
            print("Deep18 haven't been loaded.Please choose another dataset")
        elif self.filename != 'none':
            print("11")
        else:
            self.execute_javascript('alert("No file uploaded. Please try again!")')


    def fileupload_on_success(self, widget, filename):
        ''' 文件上传成功事件'''
        self.filename = filename
        print('File upload success: ' + filename)

    def fileupload_on_failed(self, widget, filename):
        ''' 文件上传失败事件'''
        print('File upload failed: ' + filename)
        self.execute_javascript('alert("No file uploaded. Please try again!")')

    def on_partition_button_click(self, button):
        """当用户点击“开始分区”按钮时触发"""
        if getattr(self, 'selected_dataset', '') == 'SIFT1B':
            if self.sample_ratio_input.get_value() == '0.01':
                self.partition_visual.attributes['src'] = 'my_res:sift_1B_4_8_3.svg'

                # 清空 partition_table 内容，但保留表头
                children_keys = list(self.partition_table.children.keys())
                for key in children_keys[1:]:  # 跳过第一个元素（表头）
                    self.partition_table.remove_child(self.partition_table.children[key])

                # 读取并填充 partition_table 的数据
                stats_file_path = "./res/sift_1B_4_8_segment_stats.txt"
                try:
                    with open(stats_file_path, 'r') as f:
                        # 跳过表头
                        header = next(f).strip()
                        print("Header:", header)

                        # 动态设置行高
                        lines = f.readlines()
                        row_count = len(lines)
                        row_height = max(20, 200 // row_count)  # 最小行高为 20px

                        partition_id = 1
                        segment_stats = []
                        for line in lines:
                            # 解析数据行
                            print("Raw line:", line)
                            start_idx, mean, variance = line.strip().split('\t')
                            start_idx = int(start_idx)
                            mean = float(mean)
                            variance = float(variance)
                            segment_stats.append((start_idx, mean, variance))

                            # 在 partition_table 中添加行
                            row = gui.TableRow(style={'height': f'{row_height}px'})
                            row.append(gui.TableItem(f"P{partition_id}"))
                            # row.append(gui.TableItem(f"{start_idx}"))
                            row.append(gui.TableItem(f"{mean:.4f}"))
                            row.append(gui.TableItem(f"{variance:.4f}"))
                            self.partition_table.append(row)
                            partition_id += 1

                    # 将 segment_stats 保存为实例变量，供 on_allocation_button_click 使用
                    self.segment_stats = segment_stats

                except FileNotFoundError:
                    print("统计信息文件未找到。")
                except Exception as e:
                    print(f"Error reading file: {e}")

                print("Images for SIFT1B dataset loaded.")
        elif getattr(self, 'selected_dataset', '') == 'Deep1B':
            print("Deep1B hasn't been loaded. Please choose another dataset.")
        elif self.filename != 'none':
            print("11")
        else:
            self.execute_javascript('alert("请至少上传一个数据集或选择一个预处理数据集。")')
        print("Partitioning process started.")

    def on_allocation_button_click(self, button):
        """当用户点击‘optimize’按钮时触发"""
        if getattr(self, 'selected_dataset', '') == 'SIFT1B':
            if self.sample_ratio_input.get_value() == '0.01':
                # 文件路径
                stats_file_path = "./res/sift_1B_4_8_segment_stats.txt"
                residual_bits_path = "./res/optimal_residual_bits.txt"
                segment_stats = []
                residual_bits_dict = {}
                max_index = 1000000000 - 1

                try:

                    with open(stats_file_path, 'r') as f:
                        header = next(f).strip()
                        for line in f:

                            start_idx, mean, variance = line.strip().split('\t')
                            segment_stats.append((int(start_idx), float(mean), float(variance)))


                    with open(residual_bits_path, 'r') as f:
                        header = next(f).strip()
                        for line in f:
                            sigma_squared, optimal_bits = line.strip().split('\t')
                            residual_bits_dict[float(sigma_squared)] = int(optimal_bits)

                    allocation_keys = list(self.allocation_table.children.keys())
                    for key in allocation_keys[1:]:
                        self.allocation_table.remove_child(self.allocation_table.children[key])


                    total_height = 150
                    row_count = len(segment_stats)
                    row_height = max(20, total_height // row_count)


                    partition_id = 1
                    prev_start_index = segment_stats[0][0]

                    for i in range(1, len(segment_stats)):
                        start_idx, mean, variance = segment_stats[i]
                        residual_bits = residual_bits_dict.get(variance, "N/A")

                        allocation_row = gui.TableRow(style={'height': f'{row_height}px'})
                        allocation_row.onclick.do(self.on_partition_row_click, partition_id)

                        allocation_row.append(gui.TableItem(f"P{partition_id}"))
                        allocation_row.append(gui.TableItem(str(prev_start_index)))
                        allocation_row.append(gui.TableItem(str(start_idx - 1)))
                        allocation_row.append(gui.TableItem(str(residual_bits)))
                        self.allocation_table.append(allocation_row)

                        prev_start_index = start_idx
                        partition_id += 1

                    last_row = gui.TableRow(style={'height': f'{row_height}px'})
                    last_row.onclick.do(self.on_partition_row_click, partition_id)
                    last_row.append(gui.TableItem(f"P{partition_id}"))
                    last_row.append(gui.TableItem(str(prev_start_index)))
                    last_row.append(gui.TableItem(str(max_index)))  # 数据集的最大索引
                    last_row.append(gui.TableItem(str(residual_bits_dict.get(segment_stats[-1][2], "N/A"))))
                    self.allocation_table.append(last_row)

                except FileNotFoundError:
                    print("文件未找到!!!!")
                except Exception as e:
                    print(f"错误错误: {e}")
        elif getattr(self, 'selected_dataset', '') == 'Deep1B':
            print("Deep1B hasn't been loaded. Please choose another dataset.")
        elif self.filename != 'none':
            print("11")
        else:
            self.execute_javascript('alert("请至少上传一个数据集或选择一个预处理数据集!!!!")')
        print("Allocation process started.")

    def on_partition_row_click(self, widget, partition_id):
        """根据点击的 Partition ID 显示相应的图片"""
        image_path = f'my_res:{partition_id}_CR-RB.svg'
        self.allocation_chart.attributes['src'] = image_path
        print(f"Displaying image: {image_path}")

    def on_benchmark_button_click(self, button):

        children_keys = list(self.benchmark_result.children.keys())
        for key in children_keys[1:]:  #跳过表头
            self.benchmark_result.remove_child(self.benchmark_result.children[key])

        bit_tuner_row = gui.TableRow()
        bit_tuner_row.append(gui.TableItem("BitTuner"))
        bit_tuner_row.append(gui.TableItem("4.245"))
        bit_tuner_row.append(gui.TableItem("34.95s"))
        bit_tuner_row.append(gui.TableItem("9.82s"))
        self.benchmark_result.append(bit_tuner_row)

        if self.lzma_checkbox.get_value():
            lzma_row = gui.TableRow()
            lzma_row.append(gui.TableItem("LZMA"))
            lzma_row.append(gui.TableItem("1.14"))
            lzma_row.append(gui.TableItem("44min54.82s"))
            lzma_row.append(gui.TableItem("2min59.36s"))
            self.benchmark_result.append(lzma_row)

        if self.lz4_checkbox.get_value():
            lz4_row = gui.TableRow()
            lz4_row.append(gui.TableItem("LZ4"))
            lz4_row.append(gui.TableItem("1.004"))
            lz4_row.append(gui.TableItem("9.506s"))
            lz4_row.append(gui.TableItem("13.416s"))
            self.benchmark_result.append(lz4_row)

        print("Benchmarking process completed.")


# 启动应用
start(BitTunerApp, address='127.0.0.1', port=8081, multiple_instance=False, enable_file_cache=False)
