function [fen] = fenshu(che,sj1,t,fan_t)
    if(size(sj1,1)<318)
        fen = -100*ones(1,5);
    else
        for i=1:318
            for j=1:318
                if sj1(i,1)==che(j,1)
                    sj1(i,2)=che(j,2);
                    sj1(i,3)=che(j,3);
                    break;
                end
            end
        end
        fen(1)=100;
        fen(2)=100;
        fen(3)=100;
        fen(4)=100;
        
        %目标1分数
        tag2=0;
        for i=1:318
            if sj1(i,2)==2
                if tag2~=2
                    fen(1)=fen(1)-1;
                end
                tag2=0;
            else 
                tag2=tag2+1;
            end
        end
        
        
        %目标2分数
        tag2=[0,0,0];j=1;
        ii=1;%ii为变化次数记录
        for i=1:318
            if sj1(i,3)==sj1(j,3)
                tag2(1,ii)=tag2(1,ii)+1;
            else
                j=i;
                ii=ii+1;
                tag2(1,ii)=tag2(1,ii)+1;
                
        
                if ii>=3
                    if tag2(1,2)~=tag2(1,1)
                        fen(2)=fen(2)-2;
                    end
                    ii=1;
                    tag2=[1,0,0];
                end
        
            end
        
        end
        
        %目标3分数
        fen(3)=fen(3)-fan_t;
        
        
        %目标4分数
        fen(4)=fen(4)-(0.01*(t-9*size(sj1,1)-72));
        
        
        %总分
        
        fen(5)=0.4*fen(1)+0.3*fen(2)+0.2*fen(3)+0.1*fen(4);
    end
end