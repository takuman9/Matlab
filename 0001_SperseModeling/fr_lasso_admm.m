function [pred,x] = fr_lasso_admm(X,y,alpha,max_iter)
% ������
T = max_iter;
% �W��
mu = alpha;

% ��������
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
    % ADMM�X�V��
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

% ���W�~���@
function y = SoftThr (x , lambda )
    % �Ƃ肠�����S��0�ɃZ�b�g
    y = zeros ( size (x));
    % �������l�𒴂���W����temp�Ɋi�[
    temp = find(x > lambda );
    % �������l�𒴂�����̂���lambda�����
    y(temp) = x( temp ) - lambda ;
    % �������l�������W����temp�Ɋi�[
    temp = find(x< -lambda );
    % �������l���������̂���lambda�����₷
    y(temp)= x(temp)+lambda ;
end