import torch
from models import VDSR

def export_vdsr_fixed_onnx(model_path, onnx_file_name="vdsr.onnx"):
    # 1. 初始化模型并加载权重
    model = VDSR(upscale_factor=4)
    state_dict = torch.load(model_path, map_location='cpu') # 兼容性写法
    model.load_state_dict(state_dict)
    model.eval() 

    # 2. 准备固定尺寸的 dummy input
    # 形状：(Batch=1, Channel=1, Height=2464, Width=1216)
    height, width = 616, 304
    dummy_input = torch.randn(1, 3, height, width)

    # 3. 导出
    print(f"正在导出固定分辨率 ({width}x{height}) 的模型...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_name,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # 注意：这里删除了 dynamic_axes，导出的模型将只接受该尺寸输入
    )
    print(f"导出完成：{onnx_file_name}")

# 执行导出
export_vdsr_fixed_onnx("./vdsr_x4_rgb.pth")