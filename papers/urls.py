from django.urls import path
from . import views
from .views import load_data, user_login, user_register, user_info, user_update, search_by_keyword, get_paper_detail, \
    get_similar_papers, get_related_papers, get_cited_papers, search

urlpatterns = [
    path('train-model', views.train_model, name='train_model'),
    path('classify-test', views.classify_test_set, name='classify_test_set'),
    path('load-data', load_data, name='load_data'),

    # 用户相关
    path("users/login", user_login, name="user_login"),
    path("users/register", user_register, name="user_register"),
    path("users", user_info, name="user_info"),
    path("users/update", user_update, name="user_update"),

    # 论文相关
    path("home", search, name="search"),
    path("search/<str:keyword>/<int:page>", search_by_keyword, name="search_by_keyword"),

    path("papers/<int:paperId>/detail", get_paper_detail, name="get_paper_detail"),
    path("papers/<int:paperId>/category/<int:page>", get_similar_papers, name="get_similar_papers"),
    path("papers/<int:paperId>/related", get_related_papers, name="get_related_papers"),
    path("papers/<int:paperId>/cited", get_cited_papers, name="get_cited_papers"),
]