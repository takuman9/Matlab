%% �ݒ����

% �V�[�h�̐ݒ�
rng = 0;

% �����ʂƃT���v���T�C�Y
n_features      = 1000; % ������
n_samples       = 100;  % �T���v����
n_nonzero_coefs = 20;   % �L���ȌW��
n_iter          = 50;   % �J��Ԃ���

% ���K���p�����[�^
alpha = 0.75;

%% �^�̏d�ݐ���
idx    = randi([0, n_features], 1,n_nonzero_coefs); % idx�F0-1000�܂ł̂��������_����20���߂�)
w      = zeros(n_features,1);                       % w  �F1000�̓����ʕ��̌W���̔�������
w(idx) = randn(n_nonzero_coefs,1);                  % w  �F�����_���Ȓl��20�̌W���ɒl������


%% ���̓f�[�^�Ɗϑ����
X = randn(n_samples,n_features); % X�F100��1000
y = X*w + randn(n_samples,1);    % y�F[100��1000]*[1000��1]�{��*[100��1]�@=�@[100��1]


%% �O���t�̕`��
ax1 = subplot(2,2,1);
ax2 = subplot(2,2,2);
stem(ax1,w);
plot(ax2,y);
title(ax1,'coeficients');
title(ax2,'value');

%% LASSO�����iADMM�j
[pred, w_]= fr_lasso_admm(X,y,alpha,n_iter);


%% �O���t�̕`��
ax2 = subplot(2,2,4);
ax3 = subplot(2,2,3);
ax4 = subplot(2,2,4);
stem(ax3,[w,w_]);
plot(ax2,y, 'Color','blue');hold on;
plot(ax2,pred,'Color','red' );hold off;
title(ax1,'coeficients');
title(ax2,'value');
title(ax3,'sperse coef');
title(ax4,'sperse value');
