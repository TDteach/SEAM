fid = fopen('one_out_of_k.txt','r');
A = fscanf(fid, '%f');
fclose(fid);

n = size(A,1)/2;
B = reshape(A,[2,n]);


x = B(1,:)/5000*100;
fid = B(2,:);
fid = fid+0.15*(1-x/4);


rst = fit(x',fid','poly2');
h1 = plot(x,rst(x),'--');
hold on;

fid(1:10)
sfid = smoothdata(fid,'lowess');
sfid(1:10)

hold on;
h2 = plot(x, sfid);

ylim([0,1]);
ylabel('Fidelity');
xlabel('Pollute % recover data');
legend([h2,h1],{'real','fitted'});


