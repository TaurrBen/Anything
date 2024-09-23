function varargout = chepaishibiegui(varargin)
% CHEPAISHIBIEGUI MATLAB code for chepaishibiegui.fig
%      CHEPAISHIBIEGUI, by itself, creates a new CHEPAISHIBIEGUI or raises the existing
%      singleton*.
%
%      H = CHEPAISHIBIEGUI returns the handle to a new CHEPAISHIBIEGUI or the handle to
%      the existing singleton*.
%
%      CHEPAISHIBIEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CHEPAISHIBIEGUI.M with the given input arguments.
%
%      CHEPAISHIBIEGUI('Property','Value',...) creates a new CHEPAISHIBIEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before chepaishibiegui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to chepaishibiegui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help chepaishibiegui


% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @chepaishibiegui_OpeningFcn, ...
    'gui_OutputFcn',  @chepaishibiegui_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before chepaishibiegui is made visible.
function chepaishibiegui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to chepaishibiegui (see VARARGIN)

% Choose default command line output for chepaishibiegui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes chepaishibiegui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = chepaishibiegui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
clc
[fname,pname,index] = uigetfile({'*.jpg';'*.bmp'},'选择图片');

str = [pname fname];
c = imread(str);
axes(handles.chepai);
imshow(c);
title('雾霾车牌')
%%%%%%% 去雾霾

%读入图像并显示
I_ori=c;
I=double(I_ori)/255;



%获取图像尺寸
[h,w,c]=size(I);

%--------------------计算图像的暗原色--------------------------------------
dark_ori=ones(h,w);
dark_extend=ones(h+8,w+8);
window=4;%扫描步长
%求出三个颜色通道的最小值
for i=1:h
    for j=1:w
        dark_extend(i+window,j+window)=min(I(i,j,:));
    end
end
%在方形区域内求出暗原色
for i=1+window:h+window
    for j=1+window:w+window
        A=dark_extend(i-window:i+window,j-window:j+window);
        dark_ori(i-window,j-window)=min(min(A));
    end
end
axes(handles.axes10);
imshow(dark_ori);
title('暗原色图像');
%--------------------计算图像的暗原色结束-----------------------------------

%估测大气光
% [dark_sort,index]=sort(dark_ori(:),'ascend');
% dark_chose=dark_sort(1:round(0.001*w*h));
% for i=1:round(0.001*w*h)
%     I_chose(i)=I(index(i));
% end
% A=max(I_chose);
A=220/255;

%估测透射率分布
w_1=0.95;
t=ones(w,h);
t=1-w_1*dark_ori/A;
t=max(min(t,1),0);
axes(handles.axes11)
imshow(t);
title('透射率图');

%------------改进透射率----------------------------------------------------
dark_ori1=min(min(min(I(:,:,:))));
dark_max1=zeros(w,h);
for i=1:h
    for j=1:w
        dark_max1(i,j)=min(I(i,j,:));
    end
end
dark_max=max(max(dark_max1(:,:)));
t1=ones(h,w);
t2=ones(h,w);
for i=1:h
    for j=1:w
        t1(i,j)=(dark_max-dark_ori1)*(A-min(I(i,j,:)));
        t2(i,j)=(dark_max-dark_ori1)*A-(min(I(i,j,:))-dark_ori1)*min(I(i,j,:));
        t(i,j)=t1(i,j)/t2(i,j);
    end
end
t=max(min(t,1),0);

%-------------改进透射率结束----------------------------------------------

%复原物体光线，得到无雾图像
t0=0.1;%透射因子下限t0
dehaze=zeros(h,w,c);
for i=1:c
    for j=1:h
        for l=1:w
            dehaze(j,l,i)=(I(j,l,i)-A)/max(t(j,l),t0)+A;
        end
    end
end
axes(handles.axes12);
imshow(dehaze);
title('去雾后图像');

imwrite(dehaze,'去雾车牌.jpg')
%%%去雾处理完毕



I = imread('去雾车牌.jpg');
I1=rgb2gray(I);
axes(handles.axes13);
imshow(I1)
title('灰度图像');
I2=edge(I1,'roberts',0.18,'both');
axes(handles.axes14);
imshow(I2)
title('边缘检测图像');
se=[1;1;1];
I3=imerode(I2,se);
axes(handles.axes15);
imshow(I3)
title('腐蚀图像');
se=strel('rectangle',[25,25]);
I4=imclose(I3,se);
axes(handles.axes16);
imshow(I4)
title('膨胀图像');
[n1, n2] = size(I4);
I4(1:round(n1/3), 1:n2) = 0;

r = floor(n1/10);
c = floor(n2/10);
I4(1:r,:)=0;
I4((9*r):n1,:)=0;
I4(:,1:c)=0;
I4(:,(c*9):n2)=0;
I5=bwareaopen(I4,200);
[y,x,z]=size(I5);
myI = double(I5);
tic
Blue_y=zeros(y,1);
for i=1:y
    for j=1:x
        if(myI(i,j,1)==1)
            Blue_y(i,1)=Blue_y(i,1)+1;
        end
    end
end
[temp MaxY] = max(Blue_y);

PY1=MaxY;
while((Blue_y(PY1,1)>=5)&&(PY1>1))
    PY1=PY1-1;
end
PY2=MaxY;
while ((Blue_y(PY2,1)>=5)&&(PY2<y))
    PY2=PY2+1;
end
IY=I(PY1:PY2,:,:);
axes(handles.axes17);
imshow(IY);
title('车牌分割');
% figure,imshow(IY);

Blue_x=zeros(1,x);
for j=1:x
    for i=PY1:PY2
        if(myI(i,j,1)==1)
            Blue_x(1,j)=Blue_x(1,j)+1;
        end
    end
end
% [tempx MaxX] = max(Blue_x);
PX1=1;
[Irow,Icol,Idisanwei] = size(I);
while((Blue_x(1,PX1)<3)&&(PX1<x))
    PX1=PX1+1;
end
PX2=x;
while ((Blue_x(1,PX2)<3)&&(PX2>PX1))
    PX2=PX2-1;
end
PX1=PX1-1;
PX2=PX2;
% dw=I(PY1:(PY2-8),PX1:PX2,:);
dw=I((PY1):(PY2),PX1:PX2,:);
t=toc;

if ((PX2-PX1)/(PY2-PY1))>4.6
    PX1=1;
    [Irow,Icol,Idisanwei] = size(I);
    while((Blue_x(1,PX1)<11)&&(PX1<x))
        PX1=PX1+1;
    end
    PX2=x;
    while ((Blue_x(1,PX2)<11)&&(PX2>PX1))
        PX2=PX2-1;
    end
    PX1=PX1-1;
    PX2=PX2;
    dw=I((PY1):(PY2),PX1:PX2,:);
end




imwrite(dw,'dw.jpg');
a=imread('dw.jpg');
b=rgb2gray(a);


g_max=double(max(max(b)));
g_min=double(min(min(b)));
T=round(g_max-(g_max-g_min)/3);
[m,n]=size(b);
d=(double(b)>=T);



h=fspecial('average',3);

d=im2bw(round(filter2(h,d)));




se=eye(2);
[m,n]=size(d);
if bwarea(d)/m/n>=0.365
    d=imerode(d,se);
elseif bwarea(d)/m/n<=0.235
    d=imdilate(d,se);
end




[dw1,dw2,dw3]=size(dw);
fanwei = round(0.0026*dw1*dw2);
d=bwareaopen(d,fanwei);
d=qiege(d);
[m,n]=size(d);
if (n/m)<4
    row = m-(n/(4.1));
    d = d(round(row):m,1:n);
end

k1=1;
k2=1;
s=sum(d);
j=1;
while j~=n
    while s(j)==0
        j=j+1;
    end
    k1=j;
    while s(j)~=0&&j<=n-1
        j=j+1;
    end
    k2=j-1;%
    if k2-k1>=round(n/6.5)%
        [val,num]=min(sum(d(:,[k1+5:k2-5])));%
        d(:,k1+num+5)=0;  %
    end
end

% 再切割
d=qiege(d);


y1=10;y2=0.25;flag=0;word1=[];
while flag==0
    [m,n]=size(d);
    left=1;wide=0;
    while sum(d(:,wide+1))~=0
        wide=wide+1;
    end
    
    if (wide<5)&&(sum(d(:,wide+1))<(10/m))
        d(:,[1:wide])=0;
        d=qiege(d);
    else
        temp=qiege(imcrop(d,[1 1 wide m]));
        [m,n]=size(temp);
        all=sum(sum(temp));
        two_thirds=sum(sum(temp([round(m/3):2*round(m/3)],:)));
        if two_thirds/all>y2
            flag=1;word1=temp;   % WORD 1
        end
        d(:,[1:wide])=0;d=qiege(d);
    end
end
[word2,d]=getword(d);
[word2row,word2col] = size(word2);
whitesizeofword2 = round(word2row*word2col/12)+1;
word2=bwareaopen(word2,whitesizeofword2);
word2=qiege(word2);
% 分割出第三个字符
[word3,d]=getword(d);
% 分割出第四个字符
[word4,d]=getword(d);
% 分割出第五个字符
[word5,d]=getword(d);
% 分割出第六个字符
[word6,d]=getword(d);
[word6row,word6col] = size(word6);
whitesizeofword6 = round(word6row*word6col/12)+1;
word6=bwareaopen(word6,whitesizeofword6);
word6=qiege(word6);
% 分割出第七个字符
[word7,d]=getword(d);

[m,n]=size(word1);

word1=imresize(word1,[32 16]);

word2=wordprocess(word2);
word3=wordprocess(word3);
word4=wordprocess(word4);
word5=wordprocess(word5);
word6=wordprocess(word6);
word7=wordprocess(word7);
axes(handles.zifu1);
imshow(word1);
axes(handles.zifu2);
imshow(word2);
axes(handles.zifu3);
imshow(word3);
axes(handles.zifu4);
imshow(word4);
axes(handles.zifu5);
imshow(word5);
axes(handles.zifu6);
imshow(word6);
axes(handles.zifu7);
imshow(word7);

imwrite(word1,'1.jpg');
imwrite(word2,'2.jpg');
imwrite(word3,'3.jpg');
imwrite(word4,'4.jpg');
imwrite(word5,'5.jpg');
imwrite(word6,'6.jpg');
imwrite(word7,'7.jpg');



%全字符模板库
char=[];
store=strcat('A','B','C','D','E','F','G','H','J','K','L','M','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','京','津','沪','渝','冀','晋','辽','吉','黑','苏','浙' ,'皖','闽','赣','鲁','豫','鄂','湘','粤','琼','川','贵','云','陕','甘','青','藏','桂','蒙','新','宁','港');
for i=1:7
    for j=1:67
        Im=eval(strcat('word',num2str(i)));
        Template=imread(strcat('example\',num2str(j),'.bmp'));
        Template=im2bw(Template);
        Differ=Im-Template;
        Compare(j)=sum(sum(abs(Differ)));
    end
    index=find(Compare==(min(Compare)));
    char=[char store(index)];
end
axes(handles.chepaijieguo);
imshow(b);
set(handles.wenzijieguo,'string',char);
%     figure(10),imshow(b),title(strcat('车牌为:',char));




function wenzijieguo_Callback(hObject, eventdata, handles)
% hObject    handle to wenzijieguo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wenzijieguo as text
%        str2double(get(hObject,'String')) returns contents of wenzijieguo as a double


% --- Executes during object creation, after setting all properties.
function wenzijieguo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wenzijieguo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
clc
close all
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
