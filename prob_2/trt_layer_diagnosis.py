#!/usr/bin/env python3
"""
TensorRT逐层诊断：静态 vs 动态模式对比
核心要求：使用完全相同的输入数据
"""
import onnx
from onnx import shape_inference
import numpy as np
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import logging
import time

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not available")

# ONNX Runtime for baseline
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if TRT_AVAILABLE else None


class TRTLayerDiagnosis:
    """TensorRT逐层诊断工具"""

    def __init__(self, model_path: str, output_dir: str = "trt_layer_diagnosis"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存输入数据路径
        self.input_data_file = os.path.join(output_dir, "shared_input_data.pkl")

        logger.info(f"加载模型: {model_path}")
        self.model = onnx.load(model_path)

        # Shape inference
        try:
            self.model = shape_inference.infer_shapes(self.model)
            logger.info("✓ Shape inference完成")
        except Exception as e:
            logger.warning(f"⚠ Shape inference失败: {e}")

        # 构建节点信息
        self._build_graph_info()

    def _build_graph_info(self):
        """构建图信息"""
        self.nodes = []
        self.node_dict = {}

        for idx, node in enumerate(self.model.graph.node):
            node_info = {
                'index': idx,
                'name': node.name if node.name else f"{node.op_type}_{idx}",
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'node': node
            }
            self.nodes.append(node_info)
            self.node_dict[node_info['name']] = node_info

        logger.info(f"模型包含 {len(self.nodes)} 个节点")

    def compute_node_depths(self):
        """计算节点深度（拓扑排序）"""
        # 图输入
        graph_inputs = {inp.name: 0 for inp in self.model.graph.input}
        initializers = {init.name: 0 for init in self.model.graph.initializer}

        tensor_depth = {**graph_inputs, **initializers}
        node_depth = {}

        max_iterations = len(self.nodes) * 2
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            progress = False

            for node_info in self.nodes:
                if node_info['name'] in node_depth:
                    continue

                # 检查所有输入是否有深度
                input_depths = []
                all_ready = True
                for inp in node_info['inputs']:
                    if inp in tensor_depth:
                        input_depths.append(tensor_depth[inp])
                    else:
                        all_ready = False
                        break

                if all_ready:
                    depth = max(input_depths) + 1 if input_depths else 0
                    node_depth[node_info['name']] = depth
                    node_info['depth'] = depth

                    # 更新输出深度
                    for out in node_info['outputs']:
                        tensor_depth[out] = depth

                    progress = True

            if not progress:
                break

        # 为未处理的节点设置默认深度
        for node_info in self.nodes:
            if 'depth' not in node_info:
                node_info['depth'] = 999

        # 排序
        self.nodes_sorted = sorted(self.nodes, key=lambda n: n['depth'])

        logger.info(f"节点深度计算完成，最大深度: {max([n['depth'] for n in self.nodes if n['depth'] < 999])}")

        return node_depth

    def save_input_data(self, input_data: Dict[str, np.ndarray]):
        """保存输入数据（确保静态和动态使用相同数据）"""
        logger.info(f"保存输入数据到: {self.input_data_file}")

        # 保存pickle格式
        with open(self.input_data_file, 'wb') as f:
            pickle.dump(input_data, f)

        # 同时保存为numpy格式（方便检查）
        npz_file = self.input_data_file.replace('.pkl', '.npz')
        np.savez(npz_file, **input_data)

        # 保存统计信息
        stats = {}
        for name, data in input_data.items():
            stats[name] = {
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max())
            }

        stats_file = self.input_data_file.replace('.pkl', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("✓ 输入数据已保存")

    def load_input_data(self) -> Dict[str, np.ndarray]:
        """加载输入数据（确保一致性）"""
        if not os.path.exists(self.input_data_file):
            raise FileNotFoundError(f"输入数据文件不存在: {self.input_data_file}")

        logger.info(f"加载输入数据: {self.input_data_file}")
        with open(self.input_data_file, 'rb') as f:
            input_data = pickle.load(f)

        logger.info("✓ 输入数据已加载")
        return input_data

    def extract_subgraph_to_node(self, target_node_name: str) -> str:
        """提取从输入到指定节点的子图"""
        target_node = self.node_dict.get(target_node_name)
        if not target_node:
            raise ValueError(f"节点不存在: {target_node_name}")

        # BFS收集所有依赖节点
        needed_nodes = set()
        queue = [target_node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current['name'] in visited:
                continue
            visited.add(current['name'])
            needed_nodes.add(current['name'])

            # 找到产生输入的节点
            for inp in current['inputs']:
                for node_info in self.nodes:
                    if inp in node_info['outputs'] and node_info['name'] not in visited:
                        queue.append(node_info)

        # 收集节点（保持原始顺序）
        subgraph_nodes = [n['node'] for n in self.nodes if n['name'] in needed_nodes]

        # 创建子图，从原始模型的 value_info 中获取输出类型信息
        # 构建一个张量名到类型信息的映射
        tensor_type_map = {}
        for vi in self.model.graph.value_info:
            tensor_type_map[vi.name] = vi.type

        # 也包含原始模型的输出
        for out_vi in self.model.graph.output:
            tensor_type_map[out_vi.name] = out_vi.type

        output_value_infos = []
        for out in target_node['outputs']:
            # 尝试从原始模型中获取类型信息
            if out in tensor_type_map:
                # 复制原始的类型信息
                vi = onnx.ValueInfoProto()
                vi.name = out
                vi.type.CopyFrom(tensor_type_map[out])
                output_value_infos.append(vi)
            else:
                # 如果找不到，创建一个空的 ValueInfo，稍后通过 shape inference 推断
                vi = onnx.ValueInfoProto()
                vi.name = out
                output_value_infos.append(vi)

        subgraph = onnx.helper.make_graph(
            subgraph_nodes,
            f"subgraph_to_{target_node_name}",
            self.model.graph.input,
            output_value_infos,
            self.model.graph.initializer
        )

        submodel = onnx.helper.make_model(subgraph)
        # Clear default opsets and manually copy from original model (protobuf compatibility)
        del submodel.opset_import[:]
        for opset in self.model.opset_import:
            op_import = submodel.opset_import.add()
            op_import.domain = opset.domain
            op_import.version = opset.version
        # Set compatible IR version for older ONNX Runtime
        submodel.ir_version = 8

        # 关键修复：使用 shape inference 来推断输出类型
        try:
            submodel = shape_inference.infer_shapes(submodel)
        except Exception as e:
            logger.warning(f"Shape inference for subgraph failed: {e}")

        # 保存
        safe_name = target_node_name.replace('/', '_')
        subgraph_file = os.path.join(
            self.output_dir,
            f"subgraph_depth{target_node['depth']:03d}_{safe_name}.onnx"
        )
        onnx.save(submodel, subgraph_file)

        return subgraph_file

    def run_onnx_runtime(self, model_path: str, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """使用ONNX Runtime运行（作为baseline）"""
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 准备输入
        feed_dict = {}
        for inp in session.get_inputs():
            if inp.name in input_data:
                feed_dict[inp.name] = input_data[inp.name]

        # 推理
        outputs = session.run(None, feed_dict)

        # 收集输出
        output_dict = {}
        for idx, out in enumerate(session.get_outputs()):
            output_dict[out.name] = outputs[idx]

        return output_dict

    def run_trt_static(self, model_path: str, input_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, np.ndarray]:
        """使用TensorRT静态模式运行（固定batch size的optimization profile）"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        # 构建引擎
        logger.info("  [静态] 构建TensorRT引擎...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)

        # 对于包含动态shape的网络，设置固定的optimization profile（静态batch）
        profile = builder.create_optimization_profile()
        has_dynamic_input = False
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape

            # 检查是否有动态维度
            if -1 in input_shape:
                has_dynamic_input = True
                actual_shape = list(input_data[input_tensor.name].shape)
                # 静态模式：min=opt=max都设置为相同的固定shape
                profile.set_shape(input_tensor.name, actual_shape, actual_shape, actual_shape)

        if has_dynamic_input:
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized engine")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        # 推理
        context = engine.create_execution_context()
        stream = cuda.Stream()

        # 如果有动态输入，设置输入shape
        if has_dynamic_input:
            for inp_name, inp_data in input_data.items():
                context.set_input_shape(inp_name, inp_data.shape)

        # 分配缓冲区
        buffers = []
        bindings = []
        output_dict = {}
        d_inputs = []
        d_outputs = []

        # 输入
        for inp_name, inp_data in input_data.items():
            d_input = cuda.mem_alloc(inp_data.nbytes)
            cuda.memcpy_htod_async(d_input, inp_data, stream)
            d_inputs.append(d_input)
            context.set_tensor_address(inp_name, int(d_input))

        # 输出
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                shape = context.get_tensor_shape(tensor_name)
                size = trt.volume(shape)
                dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

                h_output = np.empty(size, dtype=dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                d_outputs.append((tensor_name, h_output, d_output, shape))
                context.set_tensor_address(tensor_name, int(d_output))

        # 执行
        context.execute_async_v3(stream_handle=stream.handle)

        # 复制输出
        for tensor_name, h_output, d_output, shape in d_outputs:
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

        stream.synchronize()

        for tensor_name, h_output, d_output, shape in d_outputs:
            output_dict[tensor_name] = h_output.reshape(shape)
            d_output.free()

        for d_input in d_inputs:
            d_input.free()

        return output_dict

    def run_trt_dynamic(self, model_path: str, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """使用TensorRT动态模式运行"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        # 构建引擎
        logger.info("  [动态] 构建TensorRT引擎...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # 设置动态shape
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape

            # 获取实际batch size
            actual_shape = list(input_data[input_tensor.name].shape)
            batch_size = actual_shape[0]

            # 设置profile
            min_shape = [1] + actual_shape[1:]
            opt_shape = [1024] + actual_shape[1:]
            max_shape = [2048] + actual_shape[1:]

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        # 推理
        context = engine.create_execution_context()
        stream = cuda.Stream()

        # 设置输入shape
        for inp_name, inp_data in input_data.items():
            context.set_input_shape(inp_name, inp_data.shape)

        # 分配缓冲区并推理
        output_dict = {}
        d_inputs = []
        d_outputs = []

        # 输入
        for inp_name, inp_data in input_data.items():
            d_input = cuda.mem_alloc(inp_data.nbytes)
            cuda.memcpy_htod_async(d_input, inp_data, stream)
            d_inputs.append(d_input)
            context.set_tensor_address(inp_name, int(d_input))

        # 输出
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                shape = context.get_tensor_shape(tensor_name)
                size = trt.volume(shape)
                dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

                h_output = np.empty(size, dtype=dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                d_outputs.append((tensor_name, h_output, d_output, shape))
                context.set_tensor_address(tensor_name, int(d_output))

        # 执行
        context.execute_async_v3(stream_handle=stream.handle)

        # 复制输出
        for tensor_name, h_output, d_output, shape in d_outputs:
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

        stream.synchronize()

        for tensor_name, h_output, d_output, shape in d_outputs:
            output_dict[tensor_name] = h_output.reshape(shape)
            d_output.free()

        for d_input in d_inputs:
            d_input.free()

        return output_dict

    def compare_outputs(self, static_out: Dict, dynamic_out: Dict, onnx_out: Dict = None) -> Dict:
        """对比输出并计算评分"""
        comparison = {}

        for name in static_out.keys():
            if name not in dynamic_out:
                continue

            s_out = static_out[name]
            d_out = dynamic_out[name]
            o_out = onnx_out[name]
            print(f"{name}: {s_out} vs {d_out} vs {o_out}")

            diff = np.abs(s_out - d_out)
            rel_diff = diff / (np.abs(s_out) + 1e-8)

            comp = {
                'max_abs_diff': float(np.max(diff)),
                'mean_abs_diff': float(np.mean(diff)),
                'max_rel_diff': float(np.max(rel_diff)),
                'mean_rel_diff': float(np.mean(rel_diff)),
                'static_mean': float(s_out.mean()),
                'dynamic_mean': float(d_out.mean()),
                'static_std': float(s_out.std()),
                'dynamic_std': float(d_out.std())
            }

            if onnx_out and name in onnx_out:
                o_out = onnx_out[name]
                comp['onnx_mean'] = float(o_out.mean())
                comp['onnx_std'] = float(o_out.std())
                comp['static_vs_onnx_max_diff'] = float(np.max(np.abs(s_out - o_out)))
                comp['dynamic_vs_onnx_max_diff'] = float(np.max(np.abs(d_out - o_out)))

            # 计算评分 (0-100, 100表示完全一致)
            # 基于相对误差和绝对误差的组合
            max_abs_diff = comp['max_abs_diff']
            mean_rel_diff = comp['mean_rel_diff']

            # 评分公式：考虑绝对误差和相对误差
            if max_abs_diff < 1e-7:
                score = 100.0  # 几乎完全一致
            elif max_abs_diff < 1e-6:
                score = 95.0 - min(mean_rel_diff * 1000, 5.0)  # 90-95分
            elif max_abs_diff < 1e-5:
                score = 85.0 - min(mean_rel_diff * 500, 10.0)  # 75-85分
            elif max_abs_diff < 1e-4:
                score = 70.0 - min(mean_rel_diff * 200, 15.0)  # 55-70分
            elif max_abs_diff < 1e-3:
                score = 50.0 - min(mean_rel_diff * 100, 20.0)  # 30-50分
            else:
                score = max(0.0, 30.0 - np.log10(max_abs_diff + 1) * 10)  # 0-30分

            comp['similarity_score'] = round(score, 2)

            # 评级
            if score >= 95:
                comp['grade'] = 'A+'
            elif score >= 90:
                comp['grade'] = 'A'
            elif score >= 80:
                comp['grade'] = 'B'
            elif score >= 70:
                comp['grade'] = 'C'
            elif score >= 60:
                comp['grade'] = 'D'
            else:
                comp['grade'] = 'F'

            comparison[name] = comp

        return comparison

    def list_nodes(self, max_count: int = None):
        """列出所有节点信息"""
        self.compute_node_depths()

        logger.info("=" * 80)
        logger.info("模型节点列表")
        logger.info("=" * 80)

        nodes_to_show = self.nodes_sorted[:max_count] if max_count else self.nodes_sorted

        for idx, node_info in enumerate(nodes_to_show):
            logger.info(f"[{idx:4d}] Depth {node_info['depth']:3d} | {node_info['op_type']:20s} | {node_info['name']}")

        if max_count and len(self.nodes_sorted) > max_count:
            logger.info(f"\n... 还有 {len(self.nodes_sorted) - max_count} 个节点")

        logger.info(f"\n总节点数: {len(self.nodes_sorted)}")

    def diagnose(
        self,
        input_data: Dict[str, np.ndarray],
        max_depth: int = None,
        diff_threshold: float = 1e-5,
        stop_on_diff: bool = True,
        target_nodes: List[str] = None,
        node_range: Tuple[int, int] = None
    ):
        """执行逐层诊断

        Args:
            input_data: 输入数据
            max_depth: 最大检查深度
            diff_threshold: 差异阈值
            stop_on_diff: 发现差异时停止
            target_nodes: 指定要检查的节点名称列表
            node_range: 指定节点索引范围 (start, end)
        """

        logger.info("=" * 80)
        logger.info("TensorRT静态 vs 动态 逐层诊断")
        logger.info("=" * 80)

        # 保存输入数据
        self.save_input_data(input_data)

        # 计算深度
        self.compute_node_depths()

        # 准备结果
        results = []
        first_diff_node = None

        # 选择要检查的节点
        if target_nodes:
            # 指定节点名称
            nodes_to_check = [n for n in self.nodes_sorted if n['name'] in target_nodes]
            logger.info(f"检查指定的 {len(nodes_to_check)} 个节点")
        elif node_range:
            # 指定节点索引范围
            start_idx, end_idx = node_range
            nodes_to_check = self.nodes_sorted[start_idx:end_idx]
            logger.info(f"检查节点范围 [{start_idx}, {end_idx}), 共 {len(nodes_to_check)} 个节点")
        else:
            # 按深度过滤
            nodes_to_check = [n for n in self.nodes_sorted if n['depth'] < (max_depth if max_depth else 999)]
            logger.info(f"将检查 {len(nodes_to_check)} 个节点 (max_depth={max_depth if max_depth else 'unlimited'})")

        logger.info(f"差异阈值: {diff_threshold}")

        for idx, node_info in enumerate(nodes_to_check):
            logger.info(f"\n{'─' * 80}")
            logger.info(f"[{idx+1}/{len(nodes_to_check)}] 深度{node_info['depth']}: {node_info['name']}")
            logger.info(f"  类型: {node_info['op_type']}")

            try:
                # 提取子图
                subgraph_file = self.extract_subgraph_to_node(node_info['name'])
                logger.info(f"  子图: {os.path.basename(subgraph_file)}")

                # 运行ONNX Runtime (baseline)
                logger.info("  运行ONNX Runtime...")
                onnx_output = self.run_onnx_runtime(subgraph_file, input_data)

                # 运行TensorRT（如果可用）
                if TRT_AVAILABLE:
                    batch_size = list(input_data.values())[0].shape[0]

                    static_output = self.run_trt_static(subgraph_file, input_data, batch_size)
                    dynamic_output = self.run_trt_dynamic(subgraph_file, input_data)

                    # 对比
                    comparison = self.compare_outputs(static_output, dynamic_output, onnx_output)

                    # 输出结果
                    for out_name, comp in comparison.items():
                        logger.info(f"\n  输出: {out_name}")
                        logger.info(f"    相似度评分: {comp['similarity_score']:.2f}/100 (等级: {comp['grade']})")
                        logger.info(f"    静态  Mean: {comp['static_mean']:.6f}, Std: {comp['static_std']:.6f}")
                        logger.info(f"    动态  Mean: {comp['dynamic_mean']:.6f}, Std: {comp['dynamic_std']:.6f}")
                        if 'onnx_mean' in comp:
                            logger.info(f"    ONNX  Mean: {comp['onnx_mean']:.6f}, Std: {comp['onnx_std']:.6f}")
                        logger.info(f"    静态vs动态 最大差异: {comp['max_abs_diff']:.6e}")
                        logger.info(f"    静态vs动态 平均差异: {comp['mean_abs_diff']:.6e}")
                        logger.info(f"    静态vs动态 最大相对差异: {comp['max_rel_diff']:.6e}")
                        logger.info(f"    静态vs动态 平均相对差异: {comp['mean_rel_diff']:.6e}")

                        # 检查是否有显著差异
                        if comp['max_abs_diff'] > diff_threshold:
                            logger.warning(f"    ⚠️  发现显著差异！(阈值: {diff_threshold:.6e})")
                            if first_diff_node is None:
                                first_diff_node = node_info['name']

                    result = {
                        'index': idx,
                        'depth': node_info['depth'],
                        'name': node_info['name'],
                        'op_type': node_info['op_type'],
                        'subgraph_file': subgraph_file,
                        'comparison': comparison,
                        'has_significant_diff': any(c['max_abs_diff'] > diff_threshold for c in comparison.values())
                    }

                else:
                    # 只有ONNX Runtime
                    for out_name, out_tensor in onnx_output.items():
                        logger.info(f"\n  输出: {out_name}")
                        logger.info(f"    Shape: {out_tensor.shape}")
                        logger.info(f"    Mean: {out_tensor.mean():.6f}")
                        logger.info(f"    Std: {out_tensor.std():.6f}")

                    result = {
                        'index': idx,
                        'depth': node_info['depth'],
                        'name': node_info['name'],
                        'op_type': node_info['op_type'],
                        'subgraph_file': subgraph_file,
                        'onnx_outputs': {
                            name: {
                                'shape': list(tensor.shape),
                                'mean': float(tensor.mean()),
                                'std': float(tensor.std())
                            }
                            for name, tensor in onnx_output.items()
                        }
                    }

                results.append(result)

                # 如果发现差异且设置了stop_on_diff
                if stop_on_diff and first_diff_node == node_info['name']:
                    logger.warning(f"\n{'='*80}")
                    logger.warning(f"在节点 {first_diff_node} 发现第一个显著差异")
                    logger.warning(f"停止诊断，保留环境供人工分析")
                    logger.warning(f"{'='*80}")

                    # 保存当前状态
                    self._save_diff_context(node_info, subgraph_file, input_data, result)
                    break

            except Exception as e:
                logger.error(f"  ✗ 处理失败: {e}")
                import traceback
                traceback.print_exc()

                result = {
                    'index': idx,
                    'depth': node_info['depth'],
                    'name': node_info['name'],
                    'op_type': node_info['op_type'],
                    'error': str(e)
                }
                results.append(result)

        # 保存最终结果
        results_file = os.path.join(self.output_dir, "diagnosis_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"诊断完成")
        logger.info(f"结果保存在: {results_file}")
        logger.info(f"{'='*80}")

        return results

    def _save_diff_context(self, node_info, subgraph_file, input_data, result):
        """保存差异节点的完整上下文"""
        safe_name = node_info['name'].replace('/', '_')
        context_dir = os.path.join(self.output_dir, f"diff_node_{safe_name}")
        os.makedirs(context_dir, exist_ok=True)

        logger.info(f"\n保存差异节点上下文到: {context_dir}")

        # 1. 保存子图模型
        import shutil
        shutil.copy(subgraph_file, os.path.join(context_dir, "subgraph.onnx"))

        # 2. 保存输入数据
        with open(os.path.join(context_dir, "input_data.pkl"), 'wb') as f:
            pickle.dump(input_data, f)

        # 3. 保存对比结果
        with open(os.path.join(context_dir, "comparison.json"), 'w') as f:
            json.dump(result, f, indent=2)

        # 4. 生成README
        readme = f"""# 差异节点分析环境

## 节点信息
- 名称: {node_info['name']}
- 类型: {node_info['op_type']}
- 深度: {node_info['depth']}

## 文件说明
- `subgraph.onnx`: 从输入到该节点的子图模型
- `input_data.pkl`: 完全相同的输入数据
- `comparison.json`: 静态vs动态的对比结果

## 重现步骤

```python
import pickle
import numpy as np

# 加载输入数据
with open('input_data.pkl', 'rb') as f:
    input_data = pickle.load(f)

# 使用TensorRT静态模式
static_output = run_trt_static('subgraph.onnx', input_data, batch_size=32)

# 使用TensorRT动态模式
dynamic_output = run_trt_dynamic('subgraph.onnx', input_data)

# 对比
diff = np.abs(static_output - dynamic_output)
print(f"Max diff: {{np.max(diff)}}")
```

## 下一步调查
1. 检查该节点的TensorRT kernel选择
2. 分析静态和动态模式的优化策略差异
3. 验证是否为数值精度问题
4. 检查前置节点的累积误差
"""

        with open(os.path.join(context_dir, "README.md"), 'w') as f:
            f.write(readme)

        logger.info("✓ 差异节点上下文已保存")


def prepare_input_data(model_path: str, batch_size: int = 32, seed: int = 42) -> Dict[str, np.ndarray]:
    """准备输入数据（固定随机种子确保一致性）"""
    np.random.seed(seed)
    logger.info(f"准备输入数据 (batch_size={batch_size}, seed={seed})")

    model = onnx.load(model_path)
    input_data = {}

    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(batch_size)

        data = np.random.randn(*shape).astype(np.float32)
        input_data[inp.name] = data
        logger.info(f"  {inp.name}: {shape}")

    return input_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TensorRT逐层诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 列出所有节点
  python trt_layer_diagnosis.py --model gpu_org.onnx --list-nodes

  # 列出前50个节点
  python trt_layer_diagnosis.py --model gpu_org.onnx --list-nodes --list-count 50

  # 检查所有深度<=5的节点
  python trt_layer_diagnosis.py --model gpu_org.onnx --max-depth 5

  # 检查索引范围 [10, 20) 的节点
  python trt_layer_diagnosis.py --model gpu_org.onnx --node-range 10 20

  # 检查指定节点
  python trt_layer_diagnosis.py --model gpu_org.onnx --target-nodes "dense/concat_1" "Shape__1165"
        """
    )

    # 基本参数
    parser.add_argument('--model', default='gpu_org.onnx', help='ONNX模型路径')
    parser.add_argument('--output-dir', default='trt_layer_diagnosis', help='输出目录')

    # 节点选择参数（互斥）
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument('--list-nodes', action='store_true', help='列出所有节点并退出')
    node_group.add_argument('--max-depth', type=int, default=None, help='最大检查深度')
    node_group.add_argument('--node-range', type=int, nargs=2, metavar=('START', 'END'),
                            help='检查节点索引范围 [START, END)')
    node_group.add_argument('--target-nodes', type=str, nargs='+',
                            help='指定要检查的节点名称列表')

    # 列出节点时的参数
    parser.add_argument('--list-count', type=int, default=None,
                        help='列出节点时最多显示的数量（配合--list-nodes使用）')

    # 诊断参数
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--diff-threshold', type=float, default=1e-5, help='差异阈值')
    parser.add_argument('--stop-on-diff', action='store_true', help='发现差异时停止')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 创建诊断器
    diagnoser = TRTLayerDiagnosis(args.model, args.output_dir)

    # 如果只是列出节点
    if args.list_nodes:
        diagnoser.list_nodes(max_count=args.list_count)
        exit(0)

    if not TRT_AVAILABLE:
        logger.error("TensorRT不可用，只能运行ONNX Runtime baseline测试")

    # 准备输入（固定种子）
    input_data = prepare_input_data(args.model, args.batch_size, args.seed)

    # 准备节点范围参数
    node_range = tuple(args.node_range) if args.node_range else None

    # 运行诊断
    results = diagnoser.diagnose(
        input_data,
        max_depth=args.max_depth,
        diff_threshold=args.diff_threshold,
        stop_on_diff=args.stop_on_diff,
        target_nodes=args.target_nodes,
        node_range=node_range
    )

    logger.info("\n诊断完成")
