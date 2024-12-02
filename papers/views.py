from datetime import timedelta, datetime

import jwt
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect

from linked_papers.settings import SECRET_KEY
from .models import Essay
from .models import Edge
from .pagination import fetch
from .data_loader import load_essays_into_db, load_edges_into_db
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.contrib.auth import authenticate, login
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

            # 验证用户
            user = authenticate(username=name, password=password)
            # print(f"Authenticating user with email: {name} and password: {password}, user: {user}")
            # print("User is authenticated: ", user.is_authenticated)
            if user:
                print("NICKNAME", user.nickname)
                # request.user.username = user.nickname
                # request.user.email = user.username
                # request.user.password = user.role
                login(request, user)

                # print(request.user.email)
                request.session.set_expiry(0)
                payload = {
                    "user_id": user.id,
                    "exp": datetime.utcnow() + timedelta(hours=24),  # 24 小时有效期
                    "iat": datetime.utcnow(),
                }
                token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
                return JsonResponse({
                    "message": "Login successful",
                    "result": {
                        "token": token,
                        "name": user.username,
                        "nickname": user.nickname,
                        "role": user.role
                    }
                })
                # return JsonResponse({"message": "Login successful", "user_id": user.id, "redirect_url": "/api/home"})
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
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        if User.objects.filter(email=email).exists():
            return JsonResponse({"error": "Email already exists"}, status=400)
        user = User.objects.create_user(username=name, email=email, password=password, role = "USER")
        return JsonResponse({"message": "Registration successful"})
    return JsonResponse({"error": "Method not allowed"}, status=405)

# @csrf_exempt
def user_info(request):
    """
    获取用户, 暂时弃用
    """
    if request.method == "GET":
        print(request)
        user = request.user
        print("User: ", user)
        print("User is authenticated: ", user.is_authenticated)
        # print(user.nickname)
        if user:
            return JsonResponse({
                # "user_id": user.id,
                # "username": user.email,
                "nickname": user.username,
                # "role": user.password
            })
        else:
            return JsonResponse({"error": "User not authenticated"}, status=401)
        # users = User.objects.all().values("role", "username", "nickname", "identity")
        # return JsonResponse(list(users), safe=False)
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def user_update(request):
    """
    更新用户权限
    """
    if request.method == "POST":
        data = json.loads(request.body)
        token = data.get("token")
        print("token!! ", token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("user_id")
            user = User.objects.get(id=user_id)
            user.role = "VIP"
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

@csrf_exempt
def search_by_keyword(request, keyword, page):
    if request.method == "GET":
        """
        根据关键词分页检索论文
        """
        data = fetch(essayId=None, keyword=keyword, page=page, per_page=16)  # 调用分页功能
        return JsonResponse(data)
    return JsonResponse("Hello World!")

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
    # for k in Edge.objects.all().values():
    #     print(k)
    """
    获取引用论文
    """
    try:
        # 获取引用关系
        cited_edges = Edge.objects.filter(essay_id=paperId)
        # print("cited_edges: ", cited_edges)
        cited_paper_ids = [edge.cited_id for edge in cited_edges]
        print("cited_paper_ids: ", cited_paper_ids)

        # 获取被引用论文的信息
        cited_papers = Essay.objects.filter(id__in=cited_paper_ids)
        # print("cited_papers1: ", cited_papers)
        data = [{"id": paper.id, "title": paper.title, "abstract": paper.abstract, "year": paper.year, "category": paper.category} for paper in cited_papers]

        return JsonResponse(data, safe=False)
    except Essay.DoesNotExist:
        return JsonResponse({"error": "Paper not found"}, status=404)