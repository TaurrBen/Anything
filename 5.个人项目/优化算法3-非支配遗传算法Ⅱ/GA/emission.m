function E=emission(KE,n,k)
DE=0;
for i=1:n
    DE = DE+KE(k(i),i);
end
E = DE;