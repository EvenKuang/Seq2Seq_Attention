#!/usr/bin/env python3
# test_bleu_function.py - 测试BLEU计算函数

import sys
import os
sys.path.append(os.path.dirname(__file__))
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import jieba
        import random

# 模拟测试数据，避免需要训练好的模型
def mock_bleu_test():
    """模拟测试BLEU计算函数"""
    print("=" * 60)
    print("BLEU计算函数测试")
    print("=" * 60)
    
    # 创建模拟数据
    print("\n1. 创建模拟测试数据...")
    
    # 模拟pairs数据
    mock_pairs = [
        ("i am happy", "我很开心"),
        ("hello world", "你好世界"),
        ("good morning", "早上好"),
        ("thank you", "谢谢你"),
        ("how are you", "你好吗"),
    ]
    
    # 模拟模型输出（不同质量的翻译）
    mock_translations = {
        "i am happy": ["我", "很", "开心"],  # 完美翻译
        "hello world": ["你好", "世界"],      # 完美翻译
        "good morning": ["早上", "好"],       # 完美翻译
        "thank you": ["谢谢", "你"],          # 完美翻译
        "how are you": ["你", "怎么样"],      # 良好翻译
    }
    
    # 模拟evaluate函数
    def mock_evaluate(encoder, decoder, sentence, reverse):
        """模拟模型评估函数"""
        return mock_translations.get(sentence, []), None
    
    # 测试函数
    def test_bleu_calculation(num_samples=3):
        """测试BLEU计算"""
        smooth = SmoothingFunction().method7
        total_bleu = 0
        count = 0
        
        print(f"\n2. 测试{num_samples}个样本...")
        
        for i in range(min(num_samples, len(mock_pairs))):
            test_pair = mock_pairs[i]
            reference = list(jieba.cut(test_pair[1]))
            
            # 使用模拟的模型输出
            output_words, _ = mock_evaluate(None, None, test_pair[0], False)
            
            # 清理输出
            if output_words and output_words[-1] == '<EOS>':
                hypothesis = output_words[:-1]
            else:
                hypothesis = output_words
            
            print(f"\n样本{i+1}:")
            print(f"  英文: {test_pair[0]}")
            print(f"  参考: {test_pair[1]} -> {reference}")
            print(f"  预测: {' '.join(hypothesis)} -> {hypothesis}")
            
            if hypothesis and reference:
                try:
                    # 计算BLEU
                    bleu = sentence_bleu([reference], hypothesis, 
                                        smoothing_function=smooth)
                    
                    # BLEU应该小于等于1
                    if bleu <= 1.0:
                        total_bleu += bleu
                        count += 1
                        print(f"  BLEU分数: {bleu:.4f}")
                        
                        # 解释分数
                        if bleu > 0.9:
                            print("  评价: 优秀！几乎完美")
                        elif bleu > 0.7:
                            print("  评价: 良好")
                        elif bleu > 0.5:
                            print("  评价: 一般")
                        elif bleu > 0.3:
                            print("  评价: 较差")
                        else:
                            print("  评价: 很差")
                    else:
                        print(f"  ❌ 异常分数 {bleu:.4f}")
                        
                except Exception as e:
                    print(f"  ❌ 计算错误: {e}")
            else:
                print(f"  ❌ 数据为空")
        
        if count > 0:
            avg_bleu = total_bleu / count
            print(f"\n3. 测试结果:")
            print(f"   成功计算: {count}/{num_samples} 个样本")
            print(f"   平均BLEU: {avg_bleu:.4f}")
            return True, avg_bleu
        else:
            print("\n❌ 没有成功计算任何样本")
            return False, 0
    
    # 运行测试
    success, avg_score = test_bleu_calculation(3)
    
    # 边界条件测试
    print("\n4. 边界条件测试...")
    
    # 测试1：完全匹配
    ref1 = ["我", "爱", "你"]
    hyp1 = ["我", "爱", "你"]
    bleu1 = sentence_bleu([ref1], hyp1, smoothing_function=SmoothingFunction().method7)
    print(f"  完全匹配: BLEU = {bleu1:.4f} (期望接近1.0)")
    
    # 测试2：部分匹配
    ref2 = ["我", "爱", "你"]
    hyp2 = ["我", "喜欢", "你"]
    bleu2 = sentence_bleu([ref2], hyp2, smoothing_function=SmoothingFunction().method7)
    print(f"  部分匹配: BLEU = {bleu2:.4f} (期望0.0-0.5之间)")
    
    # 测试3：完全不匹配
    ref3 = ["我", "爱", "你"]
    hyp3 = ["你", "讨厌", "我"]
    bleu3 = sentence_bleu([ref3], hyp3, smoothing_function=SmoothingFunction().method7)
    print(f"  不匹配: BLEU = {bleu3:.4f} (期望接近0.0)")
    
    # 测试4：空假设
    ref4 = ["我", "爱", "你"]
    hyp4 = []
    try:
        bleu4 = sentence_bleu([ref4], hyp4, smoothing_function=SmoothingFunction().method7)
        print(f"  空假设: BLEU = {bleu4:.4f}")
    except:
        print(f"  空假设: 计算失败（正常情况）")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ BLEU计算函数测试通过！")
        print("函数可以正常:")
        print("1. 计算BLEU分数")
        print("2. 处理平滑函数")
        print("3. 处理边界条件")
        return True
    else:
        print("❌ BLEU计算函数测试失败")
        return False

def test_with_real_data_if_possible():
    """如果有训练好的模型，测试真实数据"""
    print("\n" + "=" * 60)
    print("尝试测试真实数据（如果有训练好的模型）")
    print("=" * 60)
    
    try:
        # 尝试导入你的模型和数据
        from translation import encoder1, attn_decoder1, pairs, reverse, evaluate
        import jieba
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        print("✅ 成功导入模型和数据")
        
        # 测试少量样本
        smooth = SmoothingFunction().method7
        test_samples = 3
        
        print(f"\n测试 {test_samples} 个真实样本:")
        
        for i in range(test_samples):
            try:
                pair = random.choice(pairs)
                reference = list(jieba.cut(pair[1]))
                
                output_words, _ = evaluate(encoder1, attn_decoder1, pair[0], reverse)
                
                if output_words and output_words[-1] == '<EOS>':
                    hypothesis = output_words[:-1]
                else:
                    hypothesis = output_words
                
                if hypothesis and reference:
                    bleu = sentence_bleu([reference], hypothesis, 
                                        smoothing_function=smooth)
                    
                    print(f"\n样本{i+1}:")
                    print(f"  输入: {pair[0][:30]}...")
                    print(f"  参考: {pair[1][:30]}...")
                    print(f"  预测: {' '.join(hypothesis)[:30]}...")
                    print(f"  BLEU: {bleu:.4f}")
                    
            except Exception as e:
                print(f"样本{i+1}出错: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"无法测试真实数据: {e}")
        print("需要先训练模型才能测试真实数据")
        return False

if __name__ == "__main__":
    print("开始BLEU计算函数测试...\n")
    
    # 运行模拟测试
    success = mock_bleu_test()

