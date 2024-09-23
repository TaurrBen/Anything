
function word=wordrec(xx)
% clear
% clc
load bp net;
xx=im2bw(xx);xx=double(xx(:));  % 使用阈值将图像转换为二进制图像
a=sim(net,xx);  % 归一划为： 32*16,则 xx=512*1;
[val,num]=max(a);
if num<=26
    word=char(double('A')+num-1);
elseif num<=36
    word=char(double('0')+num-1-26);
else
    switch num
        case 37
            word='京';
        case 38
            word='津';
        case 39
            word='沪';
        case 40
            word='渝';
        case 41
            word='港';
        case 42
            word='澳';
        case 43
            word='吉';
        case 44
            word='辽';
        case 45
            word='鲁';
        case 46
            word='豫';
        case 47
            word='冀';
        case 48
            word='鄂';
        case 49
            word='湘';
        case 50
            word='晋';
        case 51
            word='青';
        case 52
            word='皖';
        case 53
            word='苏';
        case 54
            word='赣';
        case 55
            word='浙';
        case 56
            word='闽';
        case 57
            word='粤';
        case 58
            word='琼';
        case 59
            word='台';
        case 60
            word='陕';
        case 61
            word='甘';
        case 62
            word='云';
        case 63
            word='川';
        case 64
            word='贵';
        case 65
            word='黑';
        case 66
            word='藏';
        case 67
            word='蒙';
        case 68
            word='桂';
        case 69
            word='新';
        case 70
            word='宁';
    end
end