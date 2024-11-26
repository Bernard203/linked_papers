from papers.models import Essay, Edge

# 查询数据库中论文数量
print(Essay.objects.count())

# 查询特定论文
essay = Essay.objects.first()
print(essay.title, essay.abstract, essay.year)

# 查询引用关系
print(Edge.objects.count())
