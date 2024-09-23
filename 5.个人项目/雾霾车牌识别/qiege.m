
function [e] = qiege(d)%  切割出最小范围
[m,n]=size(d);
top=1;bottom=m;left=1;right=n;
while sum(d(top,:))==0&&top<=m
    top=top+1;
end
while sum(d(bottom,:))==0&&bottom>=1
    bottom=bottom-1;
end
while sum(d(:,left))==0&&left<=n
    left=left+1;
end
while sum(d(:,right))==0&&right>=1
    right=right-1;
end
dd=right-left;
hh=bottom-top;
e=imcrop(d,[left top dd hh]);
% I2=imcrop(I,[a b c d]);
%利用裁剪函数裁剪图像， 其中，（a,b）表示裁剪后左上角像素在原图像中的位置；
%c表示裁剪后图像的宽，d表示裁剪后图像的高