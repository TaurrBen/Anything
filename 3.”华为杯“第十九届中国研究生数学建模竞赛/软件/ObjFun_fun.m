%% Author
% Author : Federico Giai Pron (federico.giaipron@gmail.com)
% Mail   : federico.giaipron@gmail.com
%% Objective function
function [fen ObjFun] = ObjFun_fun(X, Data)
switch Data.Function
    case 1
        ObjFun = +(X-27)^2+25;
    case 2
        ObjFun = +(X(1)-50)^2+(X(2)-25)^2+100;
    case 3
%         fen = model1(X');
        fen = model2(X');
        ObjFun = fen(5);
    case 4
        ObjFun = -(X(1)-50)^2-(X(2)-25)^2-100;
end
end