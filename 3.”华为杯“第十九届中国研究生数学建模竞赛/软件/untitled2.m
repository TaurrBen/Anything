 a=0;
for i=1:100
    i
    b=size(model1(),1);
    if(b~=318)
        a=a+1
    end
end