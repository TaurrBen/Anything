from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import ArticlePost
import time ,os
from media.model import final
from xgboost.sklearn import XGBClassifier

clf_first1=None
clf_second1=None
test_score1=None
F1_Score1=None
model_Data=None
total=None

def douploadTrainForm(request):
    myfile1 = request.FILES.get("uploadTrainFile1",None)
    myfile2 = request.FILES.get("uploadTrainFile2", None)
    print(myfile1,myfile2)
    if not myfile1:
        return HttpResponse("没有上传文件信息")
    if not myfile1.name.endswith('.csv'):
        return redirect("app:homepage")
    if not myfile2:
        return HttpResponse("没有上传文件信息")
    if not myfile2.name.endswith('.csv'):
        return redirect("app:homepage")
    filename1 = "./media/train/"+str(time.time())+myfile1.name
    destination1 = open(filename1,"wb+")
    for chunk in myfile1.chunks():
        destination1.write(chunk)
    destination1.close()

    filename2 = "./media/train/" + str(time.time()) + myfile2.name
    destination2 = open(filename2, "wb+")
    for chunk in myfile2.chunks():
        destination2.write(chunk)
    destination2.close()

    print(filename1,filename2)
    global clf_first1
    global clf_second1
    global test_score1
    global F1_Score1
    global total
    clf_first1, clf_second1,test_score1,F1_Score1,total = final.modelTrain(filename1,filename2)
    print(clf_first1, clf_second1,test_score1,F1_Score1,total)
    return redirect("app:homepage")

def douploadTestForm(request):
    myfile = request.FILES.get("uploadTestFile",None)
    print(myfile)
    if not myfile:
        return HttpResponse("没有上传文件信息")
    if not myfile.name.endswith('.csv'):
        return redirect("app:homepage")
    filename = "./media/test/"+str(time.time())+myfile.name
    destination = open(filename,"wb+")
    for chunk in myfile.chunks():
        destination.write(chunk)
    destination.close()
    global model_Data
    global clf_first1
    global clf_second1
    print(clf_first1, clf_second1)
    model_Data = final.modelTest(clf_first1, clf_second1, filename)
    print(model_Data)
    return redirect("app:homepage")

def homepage(request):
    articles = ArticlePost.objects.all()
    # 需要传递给模板（templates）的对象
    global test_score1
    global F1_Score1
    global model_Data
    global total
    data = model_Data
    context = {'articles': articles,
               'data':data,
               'test_score1':test_score1,
               'F1_Score1':F1_Score1,
               'total':total,
               }
    print(context)
    # render函数：载入模板，并返回context对象
    return render(request, 'app/homepage.html', context)

def downloadModel1(request):
    if os.path.exists('./media/model/result/final_xgboost_clf.pkl'):
        file = open('./media/model/result/final_xgboost_clf.pkl', 'rb')
        response = HttpResponse(file)
        response['Content-Type'] = 'application/octet-stream'  # 设置头信息，告诉浏览器这是个文件
        response['Content-Disposition'] = 'attachment;filename="final_xgboost_clf.pkl"'
        return response
    return HttpResponse("训练未完成，请稍等")

def downloadModel2(request):
    if os.path.exists('./media/model/result/final_xgboost_clf_2.pkl'):
        file = open('./media/model/result/final_xgboost_clf_2.pkl', 'rb')
        response = HttpResponse(file)
        response['Content-Type'] = 'application/octet-stream'  # 设置头信息，告诉浏览器这是个文件
        response['Content-Disposition'] = 'attachment;filename="final_xgboost_clf_2.pkl"'
        return response
    return HttpResponse("训练未完成，请稍等")

def downloadTestResult(request):
    if os.path.exists('./media/model/result/final_model_data.json'):
        file = open('./media/model/result/final_model_data.json', 'rb')
        response = HttpResponse(file)
        response['Content-Type'] = 'application/octet-stream'  # 设置头信息，告诉浏览器这是个文件
        response['Content-Disposition'] = 'attachment;filename="final_model_data.json"'
        return response
    return HttpResponse("测试未完成，请稍等")
