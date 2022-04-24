# Fourth_Seq2Seq
项目描述：
1-基于新闻内容生成标题，是seq2seq任务 
2-训练集由10万条样本组成，验证集由104个样本组成，二者组成形式为title+content  

关键步骤： 
1-使用transformer模型，为充分利用soft attention结构，在加载数据时，将训练集中每段新闻的title转换为output和gold，output和gold差一位。训练阶段，使用content对应向量+output作为模型输入，输出结果与gold计算loss来调整模型。 
2-验证阶段，将输入文本传入模型，得到输出的数字向量，转换为对应字符，标题生成效果良好。
