function d=wordprocess(d)
[m,n]=size(d);
%top 1/3, bottom 1/3
for i=1:round(m/3)
    if sum(sum(d([i:i+0],:)))==0
        ii=i;d([1:ii],:)=0;
    end
end
for i=m:-1:2*round(m/3)
    if sum(sum(d([i-0:i],:)))==0
        ii=i;d([ii:m],:)=0;
    end
end
if n~=1
    d=qiege(d);
end
% d=.. 这个可以通过训练过程设置大小
d=imresize(d,[32 16]); % 商用系统程序中归 一划大小为：32*16
% d=imresize(d,[40 20]);