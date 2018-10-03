function [pred,x] = fr_lasso_admm(X,y,alpha,max_iter)
% 反復回数
T = max_iter;
% 係数
mu = alpha;

% 初期条件
[~,n] = size(X);
x = zeros(n,1); % coeficients
z = zeros(n,1); % prediction 
u = zeros(n,1); % 

X1 = X' * inv(X*X');
X2 = eye(n,n)-X1*X;

ax1 = subplot(2,2,3);
stem(ax1,x);
title(ax1,'coeficients');
ax2 = subplot(2,2,4);
plot(ax2,X*x);
title(ax2,'value');

for t = 1:T
    % ADMM更新式
    x = X1 * y + X2*(z - u);
    z = SoftThr(x + u, 1/mu);
    u = u + x - z;
    stem(ax1,x,'Color','red');
    plot(ax2,X*z,'Color','red');
    title(ax2,t);
%     pause(0.15);
    drawnow;
end
pred = X*z;
end

% 座標降下法
function y = SoftThr (x , lambda )
    % とりあえず全て0にセット
    y = zeros ( size (x));
    % しきい値を超える集合をtempに格納
    temp = find(x > lambda );
    % しきい値を超えるものだけlambda分削る
    y(temp) = x( temp ) - lambda ;
    % しきい値を下回る集合をtempに格納
    temp = find(x< -lambda );
    % しきい値を下回るものだけlambda分増やす
    y(temp)= x(temp)+lambda ;
end