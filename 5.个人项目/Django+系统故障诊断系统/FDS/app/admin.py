from django.contrib import admin
# 导入ArticlerPost
from .models import ArticlePost

# 注册ArticlePost到admin中
admin.site.register(ArticlePost)