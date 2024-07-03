function smooth_cxx = spec2(m,x)

N=length(x);

Ixx=abs(fft(x)).^2;
smooth_cxx=zeros(N);

for k = 1:N
    if k<=m
         smooth_cxx(k) = 1/(k+m)*sum(Ixx(1:k+m));
    elseif k>=N-m
        smooth_cxx(k) = 1/(k+m)*sum(Ixx(k-m:N));
    else
        smooth_cxx(k) = 1/(2*m+1)*sum(Ixx(k-m:k+m));
    end
end

end