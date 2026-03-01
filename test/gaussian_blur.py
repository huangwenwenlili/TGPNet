from PIL import Image, ImageFilter
import numpy as np

def apply_gaussian_blur(input_path, output_path, radius=2):
    """
    对输入图片应用高斯模糊并保存结果

    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        radius (int): 模糊半径，值越大越模糊(默认2)
    """
    try:
        # 打开图片文件
        original_image = Image.open(input_path)
        print(f"成功加载图片: {input_path}")

        # 应用高斯模糊滤镜
        blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius))
        print(f"应用高斯模糊(半径={radius})")

        # 保存处理后的图片
        blurred_image.save(output_path)
        print(f"结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")



def apply_gaussian_blur_np(input_path, output_path, radius=2):
    """
    对输入图片应用高斯模糊并保存结果

    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        radius (int): 模糊半径，值越大越模糊(默认2)
    """
    try:
        # 打开图片文件
        original_image = Image.open(input_path)
        print(f"成功加载图片: {input_path}")
        orig_np = np.array(original_image)
        orig_img2 = Image.fromarray(orig_np)

        # 应用高斯模糊滤镜
        blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius))
        blurred_image1 = orig_img2.filter(ImageFilter.GaussianBlur(radius))
        print(f"应用高斯模糊(半径={radius})")

        blurred_patch = np.array(blurred_image1)
        blurred_patch = np.clip(blurred_patch, 0, 255).astype(np.uint8)
        blurred_patch = Image.fromarray(blurred_patch)


        # 保存处理后的图片
        blurred_image.save(output_path)
        blurred_image1.save("output_blurred5_1.jpg")
        blurred_patch.save("output_blurred5_2.jpg")
        print(f"结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")



if __name__ == "__main__":
    # 示例用法
    input_image = "/data/huang/datasets/cloud/clound-wy/RICE2/label/140.png"  # 替换为你的输入图片路径
    output_image = "output_blurred5-1.jpg"  # 输出图片路径

    # 调用模糊处理函数
    apply_gaussian_blur_np(input_image, output_image, radius=5)
