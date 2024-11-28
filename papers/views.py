from django.forms import model_to_dict
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from .models import Essay
from .pagination import fetch
from .data_loader import load_essays_into_db, load_edges_into_db
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.contrib.auth import authenticate
import json

def train_model(request):
    """训练分类模型"""
    Essay.train_classifier()
    return JsonResponse({'message': 'Classification trained successfully！'})

def classify_test_set(request):
    """分类测试集论文"""
    Essay.classify_test_set()
    return JsonResponse({'message': 'Test set classification complete！'})

def load_data(request):
    """
    调用数据加载函数，将论文和引用关系加载到数据库中
    """
    try:
        load_essays_into_db()
        load_edges_into_db()
        return JsonResponse({"message": "LOAD SUCCESS！"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def user_login(request):
    """
    用户登录，验证 email 和 password
    """
    if request.method == "POST":
        try:
            # 尝试从查询参数获取
            name = request.GET.get("email")
            password = request.GET.get("password")

            # 如果查询参数为空，尝试解析 JSON 请求体
            if not name or not password:
                data = json.loads(request.body)
                name = data.get("email")
                password = data.get("password")

            print(f"Authenticating user with email: {name} and password: {password}")
            # 验证用户
            user = authenticate(username=name, password=password)
            if user:
                return JsonResponse({"message": "Login successful", "result": model_to_dict(user), "redirect_url": "/api/home"})
                # return HttpResponseRedirect('/api/home')
                # return JsonResponse({"message": "Login successful", "user_id": user.id, "role": user.identity})
            else:
                return JsonResponse({"error": "Invalid email or password"}, status=401)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request format"}, status=400)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def user_register(request):
    """
    用户注册
    """
    if request.method == "OPTIONS":
        response = JsonResponse({"message": "Options request successful"})
        response["Allow"] = "POST, OPTIONS"
        return response

    if request.method == "POST":
        data = json.loads(request.body)
        name = data.get("email")
        nickname = data.get("nickname")
        password = data.get("password")
        if User.objects.filter(username=name).exists():
            return JsonResponse({"error": "Email already exists"}, status=400)
        user = User.objects.create_user(username=name, nickname=nickname, password=password)
        return JsonResponse({"message": "Registration successful", "user_id": user.id})
    return JsonResponse({"error": "Method not allowed"}, status=405)


def user_info(request):
    """
    获取用户列表
    """
    if request.method == "GET":
        users = User.objects.all().values("id", "username", "email", "identity")
        return JsonResponse(list(users), safe=False)
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def user_update(request):
    """
    更新用户权限
    """
    if request.method == "POST":
        data = json.loads(request.body)
        user_id = data.get("user_id")
        role = data.get("role")
        try:
            user = User.objects.get(id=user_id)
            user.identity = role
            user.save()
            return JsonResponse({"message": "User role updated successfully"})
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def search(request):
    """
    分页检索论文
    """
    return JsonResponse({"message": "Search successful"})

def search_by_keyword(request, keyword, page):
    """
    根据关键词分页检索论文
    """
    data = fetch(essayId=None, keyword=keyword, page=page, per_page=16)  # 调用分页功能
    return JsonResponse(data)
    # return JsonResponse("Hello World!")

def get_paper_detail(request, paperId):
    """
    根据论文 ID 获取详情
    """
    try:
        paper = Essay.objects.get(id=paperId)
        return JsonResponse({
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "category": paper.category
        })
    except Essay.DoesNotExist:
        return JsonResponse({"error": "Paper not found"}, status=404)

def get_similar_papers(request, paperId, page):
    """
    根据论文 ID 获取同类论文
    """
    data = fetch(essayId=paperId, keyword=None, page=page)
    return JsonResponse(data)

def get_related_papers(request, paperId):
    """
    获取相关论文（可以根据某种逻辑实现）
    """
    try:
        target_paper = Essay.objects.get(id=paperId)
        related_papers = Essay.objects.filter(category=target_paper.category).exclude(id=paperId)[:5]
        data = [{"id": paper.id, "title": paper.title} for paper in related_papers]
        return JsonResponse(data, safe=False)
    except Essay.DoesNotExist:
        return JsonResponse({"error": "Paper not found"}, status=404)

def get_cited_papers(request, paperId):
    """
    获取引用论文
    """
    try:
        paper = Essay.objects.get(id=paperId)
        cited_papers = paper.cited_by.all()  # 假设 `cited_by` 是 Edge 模型中定义的 related_name
        data = [{"id": cited.id, "title": cited.title} for cited in cited_papers]
        return JsonResponse(data, safe=False)
    except Essay.DoesNotExist:
        return JsonResponse({"error": "Paper not found"}, status=404)