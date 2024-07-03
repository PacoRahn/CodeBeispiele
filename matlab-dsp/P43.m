
N1=64;

z = randn(1,N1)*sqrt(0.5);
n=0:N1-1;
x1 = sin(0.2*pi*n)+2*sin(0.4*pi*n)+sin(0.45*pi*n)+z;
omega1 = 0:1/64:1;
omega2 = 0:1/128:1;

pad = zeros(1,192);
x1 = cat(2,x1,pad)';

Ixx1=(1/N1)*abs(fft(x1)).^2;

N2=256;

z = randn(1,N2)*sqrt(0.5);
n=0:N2-1;
x2 = sin(0.2*pi*n)+2*sin(0.4*pi*n)+sin(0.45*pi*n)+z;

Ixx2=(1/N2)*abs(fft(x2)).^2;

omega = 0:1/128:1-1/128;

figure;
plot(omega,10*log10(Ixx1(1:128)));hold on;
plot(omega,10*log10(Ixx2(1:128)));hold on;
stem([0.2 0.4 0.45],10*log10([64 128 64]));
xlabel('$\omega/\pi$','interpreter','latex');
ylabel('$10log_{10}(I_{XX}(e^(j\omega))$','interpreter','latex');
title('Periodogram of signals with different length N');
legend('N=64','N=256','true spectogram');






L=2;
input = x2;

N = length(x2);%length of signal

M = N/L;%length of segments
rectang = rectwin(M);

xn_seg = zeros(M,L);
Ixx2=zeros(M,L);

for l = 1:L %number of segments
    xn_seg(:,l) = x2(1+(l-1)*M : l*M);
    sum_window = sum(abs(rectang).^2);

    Ixx2(:,l) = 1/sum_window * abs(fft(rectang.*xn_seg(:,l))).^2;

end
Cxx = sum(Ixx2,2)/L;


%pad = zeros(1,256);
%x2 = cat(2,x2,pad)';

L=4;
input = x2;

N = length(x2);%length of signal

M = N/L;%length of segments
rectang = rectwin(M);

xn_seg = zeros(M,L);
Ixx2=zeros(M,L);

for l = 1:L %number of segments
    xn_seg(:,l) = x2(1+(l-1)*M : l*M);
    sum_window = sum(abs(rectang).^2);

    Ixx2(:,l) = 1/sum_window * abs(fft(rectang.*xn_seg(:,l))).^2;

end
Cxx2 = sum(Ixx2,2)/L;



%Cxx1 = spec1(input,L,rectang);
omega = 0:1/64:1-1/64;
omega2 = 0:1/32:1-1/32;
figure;
plot(omega,10*log10(Cxx(1:64)));hold on;
plot(omega2,10*log10(Cxx2(1:32)));hold on;
stem([0.2 0.4 0.45],10*log([64 128 64]));


title('Averaged Periodograms');
xlabel('$\omega/\pi$','interpreter','latex');
ylabel('$10log_{10}(C^B_{XX}(e^(j\omega))$','interpreter','latex');
legend('Barlett with L=2','Barlett with L=4','true spectogram');


