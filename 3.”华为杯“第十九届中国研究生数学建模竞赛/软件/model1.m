function fen = model1(X)
% function sj1 = model1
%che 涂装出车口序列信息 1：序号 2：动力 3：驱动 4：随机决策变量 5：是否出库（1出库）
%sj1 总装借车口序列信息
    % 表格数据预处理
    
%che 涂装出车口序列信息 1：序号 2：动力 3：驱动 4：随机决策变量 5：是否出库（1出库）
%sj1 总装借车口序列信息
    % 表格数据预处理
    clear all;
    [che,txt]=xlsread('附件1.xlsx');
    for i=2:1:size(txt,1)
        if strcmp(txt(i,3),'燃油')==1 che(i-1,2)=1; end
        if strcmp(txt(i,3),'混动')==1 che(i-1,2)=2; end
        if strcmp(txt(i,4),'两驱')==1 che(i-1,3)=2; end
        if strcmp(txt(i,4),'四驱')==1 che(i-1,3)=4; end
    end
    che(:,5)=0;
    ttt=[1,2,3,4,5,6];
    for i=1:size(che,1)
%         che(i,4)=floor(X(i));
        %che(i,4)=Results.Xpbest(i);
        %che(i,4)=ttt(i);
        che(i,4)=floor(6*rand+1);
        %che(i,4)=4;
    end
    
    %初始变量区
    time1=[18,12,6,0,12,18];
    time2=[24,18,12,6,12,18];
    jieche=0;songche=0;
    tutu=1;
    tag=0;                    
    shijian=zeros(6,10);
    fan_shijian=zeros(1,10);
    car01=zeros(6,10);
    fan_car=zeros(1,10);
    fan_t=0;
    sj1=0;
    nn=1;
    x=0;k=0;
    flag1=0;flag2=0;
    
    %初始变量区结束
    
    for t=0:6000
        if(fan_shijian(1,10)<0 && tag>300)
            a=1;
        end
        %时间减少
        jieche(t+2)=jieche(t+1)-1;
        songche(t+2)=songche(t+1)-1;
        shijian=shijian-1;
        fan_shijian=fan_shijian-1;
        for oo=1:6
            if car01(oo,1)==0
                shijian(oo,1)=shijian(oo,1)+1;
            end
        end
        
        %计数
        if ((flag2==1) && (songche(t+2)==0)) %记录出车序列
            fan_car(1,1)=tag;
            fan_shijian(1,1)=9;
            flag2=0;
        end
        if ((flag2==2) && (songche(t+2)==0)) %记录出车序列
            if(find(sj1(:,1)==tag))
                a=1;
            end
            sj1(nn,1)=tag;
            che(tag,5)=1;
            nn=nn+1;
            x=0;
            flag2=0;
        end
        %当送车横移机到时候才把车带走
        if((flag2==2) && (x~=0)&&(songche(t+2)==(time1(1,x)/2)+1))
            car01(x,1)=0;
            shijian(x,1)=0;
        end
        %此时将车标记在送车横移机上
        if((flag2==2) && (x~=0)&&(songche(t+2)>1)&&(songche(t+2)<=(time1(1,x)/2)+1))
            songcheTable(t+1,1)=tag;
            songcheTable(t+1,2)=time1(che(tag,4));
        else
            songcheTable(t+1,:)=0;
        end
        %送车机
        if songche(t+2)<=0 && sum(car01(:,1))~=0
            x=find(shijian(:,1)==min(min(shijian(:,1))));
            if sum(x(:,1)==6) y=6; end
            if sum(x(:,1)==1) y=1; end
            if sum(x(:,1)==5) y=5; end
            if sum(x(:,1)==2) y=2; end
            if sum(x(:,1)==3) y=3; end
            if sum(x(:,1)==4) y=4; end           %判断送车送谁
            x=y;
            tag=car01(x,1);  %车辆标记
            %判断是送出还是送会返车道
%             if(tag<=size(che,1))
                if(rand>0.8&&tutu<size(che,1)-10)%此处条件可以改为对送车横移机如何选择进入反车道
                    if(fan_shijian(1,1)<=0 && fan_car(1,1)==0)
                        flag2=1;
                        fan_t=fan_t+1;
                        songche(t+2)=time2(1,x)-1;
                        car01(x,1)=0;
                        shijian(x,1)=0;
                    else
                        flag2=0;
                    end
                end
%             end
            if(flag2~=1)
                songche(t+2)=time1(1,x)-1;
                flag2=2;
                if(x==4)
                    car01(x,1)=0;
                    shijian(x,1)=0;
                    if(find(sj1(:,1)==tag))
                        a=1;
                    end
                    sj1(nn,1)=tag;
                    che(tag,5)=1;
                    nn=nn+1;
                    x=0;
                end
            end
        end
        %车辆移动！！！
        for i=1:6
            for j=2:10
                if shijian(i,j)<=0 && car01(i,j-1)==0 && car01(i,j)~=0
                    car01(i,j-1)=car01(i,j);
                    car01(i,j)=0;
                    if j==2
                        shijian(i,j-1)=-1;
                    else
                        shijian(i,j-1)=9;
                    end
                end
            end
        end
        
    
        if tutu<=(size(che,1)+1)
%             if jieche<=0 && shijian(che(tutu,4),10)<=0            %接车机
%                 jieche=time1(1,che(tutu,4));
%                 shijian(che(tutu,4),10)=9+time1(1,che(tutu,4));
%                 car01(che(tutu,4),10)=tutu;
%                 che(tutu,5)=-1;     %进入标记       
%                 tutu=tutu+1;
%             end
            %接车空闲，接车随机的车道时间为0
            if (jieche(t+2)<=0)
                %判断是否返回道有车
                if((flag1~=1) && (fan_shijian(1,10)<=0 )&& (fan_car(1,10)>0))
                    flag1=1;%1送返回道来的车
                    %判断送入哪个车道
                    if shijian(6,10)<=0 k=6;end
                    if shijian(1,10)<=0 k=1;end
                    if shijian(5,10)<=0 k=5;end
                    if shijian(2,10)<=0 k=2;end
                    if shijian(3,10)<=0 k=3;end
                    if shijian(4,10)<=0 k=4;end
                    jieche(t+2)=time2(1,k);
                    tag2=fan_car(1,10);%存放车序号
                    fan_car(1,10)=0;
                end
                if(flag1~=1)
                    flag1=2;
                    if(tutu<=(size(che,1)) && (shijian(che(tutu,4),10)<=0) && (che(tutu,4)~=-1))%设置接车状态
                        jieche(t+2)=time1(1,che(tutu,4));
                        tutu=tutu+1;
                    end
                end
            end
            if (flag1==1)&&(jieche(t+2)<0)&& (fan_shijian(1,10)<=0 )&& (fan_car(1,10)>0)
                a=1;
            end 
            if ((flag1==1)&&(jieche(t+2)<=0))
                shijian(k,10)=9;
                car01(k,10)=tag2;
                flag1=0;
            end
            if ((flag1==2)&&((jieche(t+2)==(time1(1,che(tutu-1,4))/ 2))||(che(tutu-1,4)==4)) && (shijian(che(tutu-1,4),10)<=0))%卸车
                if(find(car01(che(tutu-1,4),:)==tutu-1))
                    tutu=tutu+1;
                    if(tutu>size(che,1))
                        continue;
                    end
                end
                shijian(che(tutu-1,4),10)=9;
                car01(che(tutu-1,4),10)=tutu-1;
                che(tutu-1,5)=-1;     %进入标记
            end
        end
        %返回车移动
        for j=9:-1:1
            if fan_shijian(1,j)<=0 && fan_car(1,j+1)==0 && fan_car(1,j)~=0
                fan_car(1,j+1)=fan_car(1,j);
                fan_car(1,j)=0;
                if j==9
                    fan_shijian(1,j+1)=-1;
                else
                    fan_shijian(1,j+1)=9;
                end
            end
        end
        
        if sum(che(:,5)==1)==size(che,1)   %t结束条件
            break;
        end
        jieguo1(:,:,t+1) = shijian(:,:);
        jieguo2(:,:,t+1) = car01(:,:);
        fan_car1(t+1) =fan_car(1,10);
        fan_car10(t+1) =fan_car(1,10);
        flag_t1(t+1) = flag1;
        flag_t2(t+1) = flag2;
    end %时间t 结束！
    for i=1:t
%         fprintf("(t:%d)\r\n",i);
        jieguo1(:,:,i);
        jieguo2(:,:,i);
    end
    fen = fenshu(che,sj1,t,fan_t);
% end
end