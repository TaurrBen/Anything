function C=cost(KC,n,k)
DC=0;
for i = 1:n
    DC = DC+KC(k(i),i);
end
C=DC;