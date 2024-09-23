%   分割字符
function [word,result]=getword(d)
word=[];flag=0;y1=8;y2=0.5;
% if d==[]
%   word=[];
% else
while flag==0
    [m,n]=size(d);
    wide=0;
    %         while sum(d(:,wide+1))~=0 && wide<=n-2 %只要不等于零就一直运行 直到找到边界
    while sum(d(:,wide+1))~=0 && wide<=n-2
        wide=wide+1;
    end
    temp=qiege(imcrop(d,[1 1 wide m]));
    [m1,n1]=size(temp);
    if wide<y1 && n1/m1>y2
        d(:,[1:wide])=0;
        if sum(sum(d))~=0
            d=qiege(d);  % 切割出最小范围
        else word=[];flag=1;
        end
    else
        word=qiege(imcrop(d,[1 1 wide m]));
        d(:,[1:wide])=0;
        if sum(sum(d))~=0;
            d=qiege(d);flag=1;
        else d=[];
        end
    end
end
%end
result=d;