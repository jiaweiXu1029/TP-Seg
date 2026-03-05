import argparse
import os
from Evaluation.evaluator import Eval_thread  # 导入 Eval_thread 类
from Evaluation.dataloader import EvalDataset  # 确保导入 EvalDataset 类

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Evaluation script for segmentation models")

    # 添加命令行参数，并设置默认值
    parser.add_argument('--save_test_path_root',
                        default='/home/zqq/桌面/XJW/MDSAM-master/Breast/Breast',
                        type=str,
                        help='Path to the directory where saliency maps (predictions) are saved')

    parser.add_argument('--test_paths',
                        type=str,
                        default='7',
                        help='The test dataset paths, separated by "+"')
    # 是否进行评估
    parser.add_argument('--Evaluation',
                        default=True,
                        type=bool,
                        help='Set True for evaluation, False for no evaluation')

    parser.add_argument('--methods',
                        type=str,
                        default='',
                        help='The methods to evaluate, separated by "+"')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./',
                        help='Path for saving the result.txt')

    # 解析命令行参数
    args = parser.parse_args()

    # 如果设置了 Evaluation 为 True，执行评估
    if args.Evaluation:
        methods = args.methods.split('+')
        test_paths = args.test_paths.split('+')

        for test_path in test_paths:
            dataset_name = test_path.split('/')[0]

            for method in methods:
                # 使用 os.path.join 来确保路径拼接正确，避免多余的斜杠
                pred_dir_all = os.path.join(args.save_test_path_root, dataset_name, method)
                gt_dir_all = os.path.join('/root/autodl-tmp/data/medical/test', dataset_name, 'mask')

                try:
                    loader = EvalDataset(pred_dir_all, gt_dir_all)  # 创建数据加载器
                    thread = Eval_thread(loader, method, test_path, args.save_dir, cuda=True)
                    result = thread.run()
                    print(result)
                except FileNotFoundError as e:
                    print(f"Error: {e}. Check if the directory {pred_dir_all} exists.")
    else:
        print("Evaluation is turned off. No evaluation will be performed.")

if __name__ == "__main__":
    main()
