nonFixedF = [20   12  80  98 0;
             20   12  96  98 0;
             20   12  7   31 1;
             20   12  30  48 1;
             24   12  66  94 0;
             12   12  89  93 0;
             25   12  99  97 0;
             6    5   82  97 0;
             39.6 5.6 5   73 1;
             39.6 5.6 129 98 0;];
fixedF = [37  18.45 90   55 0;
          37  18.45 55   25 0;
          3   3     120  60 0;
          3   3     30   30 0;
          140 8     85   80 0;
          80  8     155  40 1;
          155 8     80   5  0;
          100 8     20   55 1;];
Facility = [nonFixedF;fixedF;];
textX = Facility;
DrawFacility(Facility);
f1 = F1(textX);
f2 = F2(textX);
f3 = F3(textX);
function f1 = F1(X)
    %塔吊吊钩起升速度（固定） m/min
    Vh = 0.55;
    %塔吊小车牵引速度（固定） m/min
    Vr = 44;
    %塔吊回转速度（固定）m/min
    Vw = 0.6;
    %操作者熟练程度
    alpha = 0.25;
    beta = 1;
    %标准层需求点D信息（固定）
    DFloor1 = [87     67 11.6;
               102    67 11.6;
               83.75  57 11.6;
               102.25 57 11.6;];
    DFloor2 = [49.8  37 12.9;
               64.7  37 12.9;
               46.75 25 12.9;
               65.25 25 12.9;];
    %标准层供需运输量Q（固定）
    QFloor1 = [0  2;
               0  2;
               48 0;
               48 0;];
    QFloor2 = [0  2;
               0  2;
               48 0;
               48 0;];
    %标准层供应点S信息（变动）
    SFloor1 = [X(1,3:4) 0.5;X(2,3:4) 1;];
    SFloor2 = [X(3,3:4) 0.5;X(4,3:4) 1;];
    %塔吊C信息（固定）
    C1 = X(13,3:4);
    C2 = X(14,3:4);
    
    %Floor1
    for i = 1:2 %第i个供应点
        for j = 1:4 %第j个需求点
            %吊钩垂直起升时间
            TvFloor1(i,j) = abs(SFloor1(i,3)-DFloor1(j,3))/Vh;
            TvFloor2(i,j) = abs(SFloor2(i,3)-DFloor2(j,3))/Vh;
            %变幅小车径向移动时间
            L_DandCFloor1 = distance(DFloor1(j,1:2),C1);
            L_SandCFloor1 = distance(SFloor1(i,1:2),C1);
            L_DandCFloor2 = distance(DFloor2(j,1:2),C2);
            L_SandCFloor2 = distance(SFloor2(i,1:2),C2);

            TrFloor1(i,j) = abs(L_DandCFloor1-L_SandCFloor1)/Vr;
            TrFloor2(i,j) = abs(L_DandCFloor2-L_SandCFloor2)/Vr;
            %塔吊回转时间
            L_SandDFloor1 = distance(SFloor1(i,1:2),DFloor1(j,1:2));
            L_SandDFloor2 = distance(SFloor2(i,1:2),DFloor2(j,1:2));
            TwFloor1(i,j) = acos((L_SandDFloor1^2-L_DandCFloor1^2-L_SandCFloor1^2)/(2*L_DandCFloor1*L_SandCFloor1))/Vw;
            TwFloor2(i,j) = acos((L_SandDFloor2^2-L_DandCFloor2^2-L_SandCFloor2^2)/(2*L_DandCFloor2*L_SandCFloor2))/Vw;
            %塔吊水平运动速度
            ThFloor1(i,j) = max(TwFloor1(i,j),TrFloor1(i,j)) + alpha*min(TwFloor1(i,j),TrFloor1(i,j));
            ThFloor2(i,j) = max(TwFloor2(i,j),TrFloor2(i,j)) + alpha*min(TwFloor2(i,j),TrFloor2(i,j));
            %总吊装时间
            TkFloor1(i,j) = max(ThFloor1(i,j),TvFloor1(i,j)) + beta*min(ThFloor1(i,j),TvFloor1(i,j));
            TkFloor2(i,j) = max(ThFloor2(i,j),TvFloor2(i,j)) + beta*min(ThFloor2(i,j),TvFloor2(i,j));
            f1Floor1(i,j) = TkFloor1(i,j)*QFloor1(j,i);
            f1Floor2(i,j) = TkFloor2(i,j)*QFloor2(j,i);
        end
    end
    f1 = sum(f1Floor1(:)) + sum(f1Floor2(:));
end

function f2 = F2(X)
    syms A E I O U;
    costTable = [0 U A O U U O O I U U U U U U;
                 U 0 U U A O O O I U U U U U U;
                 A U 0 U U U U U U U U U U U U;
                 O U U 0 U U U U U U U U U U U;
                 U A U U 0 U U U U U U U U U U;
                 U O U U U 0 U U U U U U U U U;
                 O O U U U U 0 U U U U U U U U;
                 O O U U U U U 0 U U U U U U U;
                 I I U U U U U U 0 U U U U U U;
                 U U U U U U U U U 0 U U U U U;
                 U U U U U U U U U U 0 U U U U;
                 U U U U U U U U U U U 0 U U U;
                 U U U U U U U U U U U U 0 U U;
                 U U U U U U U U U U U U U 0 U;
                 U U U U U U U U U U U U U U 0;];
    costTable = double(subs(costTable,[A E I O U],[243 81 27 9 3]));
    
    costIndex = [3 4 5 6 7 8 11 12 13 14];
    Distance = zeros(size(costTable));
    for i=1:size(costIndex,2)
        for j=1:size(costIndex,2)
            Distance(costIndex(i),costIndex(j)) = distance(X(i,3:4),X(j,3:4));
        end
    end
    s = triu(costTable).*Distance;
    f2 = sum(sum(s));
end

function f3 = F3(X)
    noiseE = [90 5;86.5 5;77.7 5;95 6;];
    for i=1:size(noiseE,1)
        for j=1:size(noiseE,1)
            if j == i
                continue;%
            end
            p = noiseE(i,2);
            q = noiseE(j,2);
            Distance = distance(X(p,3:4),X(q,3:4));
            if p == q
                Le(i,j) = noiseE(j,1);
            else 
                Y = 5.548*log(Distance) - 1.042;
                Le(i,j) = noiseE(j,1) - Y;
            end
            Le(i,j) = 10.^(0.1*Le(i,j));
            tq(i) = 10*log10(sum(Le(i,:)));
        end
    end
    f3 = sum(tq);
end

function DrawFacility(Facility)
    for i = 1:size(Facility,1)
        if Facility(i,5)
            width = Facility(i,2);
            height = Facility(i,1);
        else
            width = Facility(i,1);
            height = Facility(i,2);
        end
        x = Facility(i,3);
        y = Facility(i,4);
        rectangle("Position",[x - 0.5*width,y - 0.5*height,width,height]);
        if (i==13) || (i==14)
            rectangle('Position',[x - 56, y - 56, 56 * 2, 56 * 2],"EdgeColor","r",'Curvature',[1,1]);
        end
        xlim([0 200]);
        ylim([0 200]);
        text(x, y, num2str(i), 'HorizontalAlignment','center', 'VerticalAlignment','top');
    end
end

