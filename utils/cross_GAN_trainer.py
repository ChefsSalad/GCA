import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
def train_triple_gan(modelG1,modelG2,modelG3,modelD1,modelD2,modelD3,trainDataloader1, trainDataloader2, trainDataloader3,window_size1,window_size2,window_size3,y_scaler,train_x1,
                         train_x2, train_x3, train_y,val_x1,val_x2,val_x3,val_y,distill,num_epochs,
                     output_dir,device):

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5

    # 二元交叉熵【损失函数，可能会有问题
    criterion = nn.BCELoss()

    optimizerG1 = torch.optim.AdamW(modelG1.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
    optimizerG2 = torch.optim.AdamW(modelG2.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
    optimizerG3 = torch.optim.AdamW(modelG3.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))


    # 为每个优化器设置 ReduceLROnPlateau 调度器
    schedulerG1 = lr_scheduler.ReduceLROnPlateau(optimizerG1, mode='min', factor=0.1, patience=16, min_lr=1e-7)
    schedulerG2 = lr_scheduler.ReduceLROnPlateau(optimizerG2, mode='min', factor=0.1, patience=16, min_lr=1e-7)
    schedulerG3 = lr_scheduler.ReduceLROnPlateau(optimizerG3, mode='min', factor=0.1, patience=16, min_lr=1e-7)

    optimizerD1 = torch.optim.Adam(modelD1.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
    optimizerD2 = torch.optim.Adam(modelD2.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
    optimizerD3 = torch.optim.Adam(modelD3.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))

    generators=[modelG1,modelG2,modelG3]
    dataloaders=[trainDataloader1,trainDataloader2,trainDataloader3]
    window_sizes=[window_size1,window_size2,window_size3]
    optimizers=[optimizerG1,optimizerG2,optimizerG3]

    # 参数配置，需要反复调整，从而获得比较理想的实验结果


    best_epoch1 = best_epoch2 = best_epoch3 = -1  #

    # 用于记录损失值
    histG1 = np.zeros(num_epochs)
    histG2 = np.zeros(num_epochs)
    histG3 = np.zeros(num_epochs)  # 添加G3的损失记录

    histD1 = np.zeros(num_epochs)
    histD2 = np.zeros(num_epochs)
    histD3 = np.zeros(num_epochs)  # 添加D3的损失记录

    # 用于记录两个判别器对于两个生成器的单独的损失
    histD1_G1 = np.zeros(num_epochs)
    histD2_G1 = np.zeros(num_epochs)
    histD3_G1 = np.zeros(num_epochs)

    histD1_G2 = np.zeros(num_epochs)
    histD2_G2 = np.zeros(num_epochs)
    histD3_G2 = np.zeros(num_epochs)

    histD1_G3 = np.zeros(num_epochs)
    histD2_G3 = np.zeros(num_epochs)
    histD3_G3 = np.zeros(num_epochs)

    # 主要是用于记录MSE的变化，看一下变化曲线
    hist_MSE_G1 = np.zeros(num_epochs)
    hist_MSE_G2 = np.zeros(num_epochs)
    hist_MSE_G3 = np.zeros(num_epochs)

    # 用来记录训练过程中的验证集上的损失和分数
    hist_val_loss1 = np.zeros(num_epochs)
    hist_val_loss2 = np.zeros(num_epochs)
    hist_val_loss3 = np.zeros(num_epochs)

    best_mse1 = float('inf')
    best_mse2 = float('inf')
    best_mse3 = float('inf')

    best_model_state1 = None
    best_model_state2 = None
    best_model_state3 = None

    patience_counter = 0
    patience = 50
    feature_num = train_x1.shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    for epoch in range(num_epochs):

        if epoch<10:
            alpha1, alpha2, alpha3, alpha4 = 1, 0, 0, 1.0
            beta1, beta2, beta3, beta4 = 0, 1, 0, 1.0
            gamma1, gamma2, gamma3, gamma4 = 0., 0, 1, 1.0
        else:
            alpha1, alpha2, alpha3, alpha4 = 0.333, 0.333, 0.333, 1.0
            beta1, beta2, beta3, beta4 = 0.333, 0.333, 0.333, 1.0
            gamma1, gamma2, gamma3, gamma4 = 0.333, 0.333, 0.333, 1.0

        modelG1.train()
        modelG2.train()
        modelG3.train()

        lossdata_G1 = []
        lossdata_G2 = []
        lossdata_G3 = []

        lossdata_D1 = []
        lossdata_D2 = []
        lossdata_D3 = []

        lossdata_D1_G1 = []
        lossdata_D2_G1 = []
        lossdata_D3_G1 = []

        lossdata_D1_G2 = []
        lossdata_D2_G2 = []
        lossdata_D3_G2 = []

        lossdata_D1_G3 = []
        lossdata_D2_G3 = []
        lossdata_D3_G3 = []

        lossdata_MSE_G1 = []
        lossdata_MSE_G2 = []
        lossdata_MSE_G3 = []


        gap13=window_size3-window_size1
        gap23=window_size3-window_size2
        for batch_idx, (x3,y3)in enumerate(trainDataloader3):

            x3 = x3.to(device)
            y3 = y3.to(device)
            x1 =x3[:,gap13:,:]
            y1 = y3[:,gap13:,:]
            x2 = x3[:,gap23:,:]
            y2 = y3[:,gap23:,:]

            modelG1.eval()
            modelG2.eval()
            modelG3.eval()
            modelD1.train()
            modelD2.train()
            modelD3.train()

            #discriminator output for real data
            dis_real_output1 = modelD1(y1)
            dis_real_output2 = modelD2(y2)
            dis_real_output3 = modelD3(y3)

            real_labels_1 = torch.ones_like(dis_real_output1).to(device)
            real_labels_2 = torch.ones_like(dis_real_output2).to(device)
            real_labels_3 = torch.ones_like(dis_real_output3).to(device)

            # 判别器对真实数据损失
            lossD_real1 = criterion(dis_real_output1, real_labels_1)
            lossD_real2 = criterion(dis_real_output2, real_labels_2)
            lossD_real3 = criterion(dis_real_output3, real_labels_3)

            # discriminator output for fake data
            # G1生成的数据
            fake_data_temp_G1 = modelG1(x1).detach()
            # G2生成的数据
            fake_data_temp_G2 = modelG2(x2).detach()
            # G3生成的数据
            fake_data_temp_G3 = modelG3(x3).detach()


            # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
            fake_data_G1 = torch.cat([y1[:, :window_size1, :], fake_data_temp_G1.reshape(-1, 1, target_num)], axis=1)
            fake_data_G2 = torch.cat([y2[:, :window_size2, :], fake_data_temp_G2.reshape(-1, 1, target_num)], axis=1)
            fake_data_G3 = torch.cat([y3[:, :window_size3, :], fake_data_temp_G3.reshape(-1, 1, target_num)], axis=1)


            #判别器对伪造数据损失
            # 三个生成器的结果的数据对齐
            fake_data_1to2 = torch.cat([y2[:, :window_size2-window_size1, :], fake_data_G1], axis=1)
            fake_data_1to3 = torch.cat([y3[:, :window_size3-window_size1, :], fake_data_G1], axis=1)

            fake_data_2to1 = fake_data_G2[:, window_size2 - window_size1:, :]
            fake_data_2to3 = torch.cat([y3[:, :window_size3 - window_size2, :], fake_data_G2], axis=1)

            fake_data_3to1 = fake_data_G3[:, window_size3 - window_size1:, :]
            fake_data_3to2 = fake_data_G3[:, window_size3 - window_size2:, :]

            # 三个判别器，对于三个生成器伪造结果的判断
            dis_fake_outputD1_1 = modelD1(fake_data_G1)
            dis_fake_outputD1_2 = modelD1(fake_data_2to1)
            dis_fake_outputD1_3 = modelD1(fake_data_3to1)

            dis_fake_outputD2_1 = modelD2(fake_data_1to2)
            dis_fake_outputD2_2 = modelD2(fake_data_G2)
            dis_fake_outputD2_3 = modelD2(fake_data_3to2)

            dis_fake_outputD3_1 = modelD3(fake_data_1to3)
            dis_fake_outputD3_2 = modelD3(fake_data_2to3)
            dis_fake_outputD3_3 = modelD3(fake_data_G3)

            fake_labels_1 = torch.zeros_like(real_labels_1).to(device)
            fake_labels_2 = torch.zeros_like(real_labels_2).to(device)
            fake_labels_3 = torch.zeros_like(real_labels_3).to(device)


            # 三个判别器对伪造数据损失
            lossD1_fake1 = criterion(dis_fake_outputD1_1, fake_labels_1)
            lossD1_fake2 = criterion(dis_fake_outputD1_2, fake_labels_1)
            lossD1_fake3 = criterion(dis_fake_outputD1_3, fake_labels_1)

            lossD2_fake1 = criterion(dis_fake_outputD2_1, fake_labels_2)
            lossD2_fake2 = criterion(dis_fake_outputD2_2, fake_labels_2)
            lossD2_fake3 = criterion(dis_fake_outputD2_3, fake_labels_2)

            lossD3_fake1 = criterion(dis_fake_outputD3_1, fake_labels_3)
            lossD3_fake2 = criterion(dis_fake_outputD3_2, fake_labels_3)
            lossD3_fake3 = criterion(dis_fake_outputD3_3, fake_labels_3)

            # 两个判别器的总的损失和
            loss_D1 = alpha1 * lossD1_fake1 + alpha2 * lossD1_fake2 +alpha3*lossD1_fake3 +alpha4*lossD_real1

            loss_D2 = beta1 * lossD2_fake1 + beta2 * lossD2_fake2 + beta3 * lossD2_fake3+ beta4*lossD_real2

            loss_D3 = gamma1 * lossD3_fake1 + gamma2 * lossD3_fake2 + gamma3 * lossD3_fake3 + gamma4 * lossD_real3

            lossdata_D1.append(loss_D1.item())
            lossdata_D2.append(loss_D2.item())
            lossdata_D3.append(loss_D3.item())

            lossdata_D1_G1.append(lossD1_fake1.item())
            lossdata_D2_G1.append(lossD2_fake1.item())
            lossdata_D3_G1.append(lossD3_fake1.item())
            lossdata_D1_G2.append(lossD1_fake2.item())
            lossdata_D2_G2.append(lossD2_fake2.item())
            lossdata_D3_G2.append(lossD3_fake2.item())
            lossdata_D1_G3.append(lossD1_fake3.item())
            lossdata_D2_G3.append(lossD2_fake3.item())
            lossdata_D3_G3.append(lossD3_fake3.item())

            # 根据批次的奇偶性交叉训练两个GAN
            #if batch_idx% 2 == 0:
            optimizerD1.zero_grad()
            optimizerD2.zero_grad()
            optimizerD3.zero_grad()
            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()
            optimizerD1.step()
            optimizerD2.step()
            optimizerD3.step()

            '''训练生成器'''
            modelD1.eval()
            modelD2.eval()
            modelD3.eval()
            modelG1.train()
            modelG2.train()
            modelG3.train()

            fake_data_temp_G1 = modelG1(x1)
            fake_data_temp_G2 = modelG2(x2)
            fake_data_temp_G3 = modelG3(x3)

            fake_data_G1 = torch.cat([y1[:, :window_size1, :], fake_data_temp_G1.reshape(-1, 1, target_num)], axis=1)
            fake_data_G2 = torch.cat([y2[:, :window_size2, :], fake_data_temp_G2.reshape(-1, 1, target_num)], axis=1)
            fake_data_G3 = torch.cat([y3[:, :window_size3, :], fake_data_temp_G3.reshape(-1, 1, target_num)], axis=1)

            fake_data_1to2 = torch.cat([y2[:, :window_size2 - window_size1, :], fake_data_G1], axis=1)
            fake_data_1to3 = torch.cat([y3[:, :window_size3 - window_size1, :], fake_data_G1], axis=1)

            fake_data_2to1 = fake_data_G2[:, window_size2 - window_size1:, :]
            fake_data_2to3 = torch.cat([y3[:, :window_size3 - window_size2, :], fake_data_G2], axis=1)

            fake_data_3to1 = fake_data_G3[:, window_size3 - window_size1:, :]
            fake_data_3to2 = fake_data_G3[:, window_size3 - window_size2:, :]

            dis_fake_outputD1_1 = modelD1(fake_data_G1)
            dis_fake_outputD1_2 = modelD1(fake_data_2to1)
            dis_fake_outputD1_3 = modelD1(fake_data_3to1)

            dis_fake_outputD2_1 = modelD2(fake_data_1to2)
            dis_fake_outputD2_2 = modelD2(fake_data_G2)
            dis_fake_outputD2_3 = modelD2(fake_data_3to2)

            dis_fake_outputD3_1 = modelD3(fake_data_1to3)
            dis_fake_outputD3_2 = modelD3(fake_data_2to3)
            dis_fake_outputD3_3 = modelD3(fake_data_G3)

            lossG1_D1 = criterion(dis_fake_outputD1_1, real_labels_1)
            lossG1_D2 = criterion(dis_fake_outputD2_1, real_labels_2)
            lossG1_D3 = criterion(dis_fake_outputD3_1, real_labels_3)

            lossG2_D1 = criterion(dis_fake_outputD1_2, real_labels_1)
            lossG2_D2 = criterion(dis_fake_outputD2_2, real_labels_2)
            lossG2_D3 = criterion(dis_fake_outputD3_2, real_labels_3)

            lossG3_D1 = criterion(dis_fake_outputD1_3, real_labels_1)
            lossG3_D2 = criterion(dis_fake_outputD2_3, real_labels_2)
            lossG3_D3 = criterion(dis_fake_outputD3_3, real_labels_3)

            loss_G1 = alpha1 * lossG1_D1 + alpha2 * lossG1_D2 + alpha3 * lossG1_D3
            loss_G2 = beta1 * lossG2_D1 + beta2 * lossG2_D2 + beta3 * lossG2_D3
            loss_G3 = gamma1 * lossG3_D1 + gamma2 * lossG3_D2 + gamma3 * lossG3_D3

            loss_mse_G1 = F.mse_loss(fake_data_temp_G1.squeeze(), y1[:, -1, :].squeeze())
            loss_G1 += loss_mse_G1

            loss_mse_G2 = F.mse_loss(fake_data_temp_G2.squeeze(), y2[:, -1, :].squeeze())
            loss_G2 += loss_mse_G2

            loss_mse_G3 = F.mse_loss(fake_data_temp_G3.squeeze(), y3[:, -1, :].squeeze())
            loss_G3 += loss_mse_G3

            lossdata_MSE_G1.append(loss_mse_G1.item())
            lossdata_MSE_G2.append(loss_mse_G2.item())
            lossdata_MSE_G3.append(loss_mse_G3.item())

            lossdata_G1.append(loss_G1.item())
            lossdata_G2.append(loss_G2.item())
            lossdata_G3.append(loss_G3.item())

            optimizerG1.zero_grad()
            optimizerG2.zero_grad()
            optimizerG3.zero_grad()
            loss_G1.backward()
            loss_G2.backward()
            loss_G3.backward()
            optimizerG1.step()
            optimizerG2.step()
            optimizerG3.step()
        histG1[epoch] = np.mean(lossdata_G1)
        histG2[epoch] = np.mean(lossdata_G2)
        histG3[epoch] = np.mean(lossdata_G3)

        histD1[epoch] = np.mean(lossdata_D1)
        histD2[epoch] = np.mean(lossdata_D2)
        histD3[epoch] = np.mean(lossdata_D3)

        histD1_G1[epoch] = np.mean(lossdata_D1_G1)
        histD2_G1[epoch] = np.mean(lossdata_D2_G1)
        histD3_G1[epoch] = np.mean(lossdata_D3_G1)

        histD1_G2[epoch] = np.mean(lossdata_D1_G2)
        histD2_G2[epoch] = np.mean(lossdata_D2_G2)
        histD3_G2[epoch] = np.mean(lossdata_D3_G2)

        histD1_G3[epoch] = np.mean(lossdata_D1_G3)
        histD2_G3[epoch] = np.mean(lossdata_D2_G3)
        histD3_G3[epoch] = np.mean(lossdata_D3_G3)

        hist_MSE_G1[epoch] = np.mean(lossdata_MSE_G1)
        hist_MSE_G2[epoch] = np.mean(lossdata_MSE_G2)
        hist_MSE_G3[epoch] = np.mean(lossdata_MSE_G3)

        # 使用验证集对Generator的生成效果进行验证

        hist_val_loss1[epoch] = validate(modelG1, val_x1, val_y)
        hist_val_loss2[epoch] = validate(modelG2, val_x2, val_y)
        hist_val_loss3[epoch] = validate(modelG3, val_x3, val_y)

        improved = [False] * 3
        if hist_val_loss1[epoch] < best_mse1:
            best_mse1 = hist_val_loss1[epoch]
            best_model_state1 = copy.deepcopy(modelG1.state_dict())
            best_epoch1 = epoch + 1
            improved[0] = True

        if hist_val_loss2[epoch] < best_mse2:
            best_mse2 = hist_val_loss2[epoch]
            best_model_state2 = copy.deepcopy(modelG2.state_dict())
            best_epoch2 = epoch + 1
            improved[1] = True

        if hist_val_loss3[epoch] < best_mse3:
            best_mse3= hist_val_loss3[epoch]
            best_model_state3 = copy.deepcopy(modelG3.state_dict())
            best_epoch3 = epoch + 1
            improved[2] = True

        # 假设val_loss是验证集的损失
        schedulerG1.step(hist_val_loss1[epoch])
        schedulerG2.step(hist_val_loss2[epoch])
        schedulerG3.step(hist_val_loss3[epoch])

        if distill and epoch%10==0:
            losses = [hist_val_loss1[epoch], hist_val_loss2[epoch], hist_val_loss3[epoch]]
            rank = np.argsort(losses)
            do_distill(rank, generators, dataloaders, optimizers, window_sizes,device)
        if epoch % 10 == 0 :
            G_losses = [hist_val_loss1[epoch], hist_val_loss2[epoch], hist_val_loss3[epoch]]
            D_losses = [np.mean(lossdata_D1), np.mean(lossdata_D2), np.mean(lossdata_D3)]
            G_rank = np.argsort(G_losses)
            D_rank = np.argsort(D_losses)

            refine_best_models_with_real_data_v2(
                G_rank, D_rank,
                generators=[modelG1, modelG2, modelG3],
                discriminators=[modelD1, modelD2, modelD3],
                g_optimizers=[optimizerG1, optimizerG2, optimizerG3],
                d_optimizers=[optimizerD1, optimizerD2, optimizerD3],
                dataloaders=dataloaders,
                window_sizes=window_sizes,
                device_G="cuda:0",  # or "cuda:1", assign as needed
                device_D="cuda:1"
            )

        # 每个epoch结束时，打印训练过程中的损失
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Validation MSE G1: {hist_val_loss1[epoch]:.8f}, G2: {hist_val_loss2[epoch]:.8f}, G3: {hist_val_loss3[epoch]:.8f}")
        print(f"patience counter:{patience_counter}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    #visualize generator loss
    data_G1 = [histD1_G1, histD2_G1, histD3_G1, histG1]
    data_G2 = [histD1_G2, histD2_G2, histD3_G2, histG2]
    data_G3 = [histD1_G3, histD2_G3, histD3_G3, histG3]
    plot_generator_losses(data_G1, data_G2, data_G3,output_dir)
    #visualize discriminator loss
    data_D1 = [histD1_G1, histD1_G2, histD1_G3, histG1]
    data_D2 = [histD2_G1, histD2_G2, histD2_G3, histG2]
    data_D3 = [histD3_G1, histD3_G2, histD3_G3, histG3]
    plot_discriminator_losses(data_D1, data_D2, data_D3,output_dir)
    #overall G&D
    visualize_overall_loss(histG1, histG2, histG3,histD1, histD2, histD3,output_dir)
    plot_mse_loss(hist_MSE_G1[:epoch], hist_MSE_G2[:epoch], hist_MSE_G3[:epoch], hist_val_loss1[:epoch], hist_val_loss2[:epoch], hist_val_loss3[:epoch], epoch,output_dir)

    print("G1 best epoch:",best_epoch1)
    print("G2 best epoch:",best_epoch2)
    print("G3 best epoch:",best_epoch3)


    results =evaluate_best_models(modelG1, modelG2, modelG3, best_model_state1, best_model_state2, best_model_state3, train_x1,
                         train_x2, train_x3, train_y, val_x1, val_x2, val_x3, val_y,y_scaler, output_dir)
    return results





def train_triple_gan_Version0(modelG1,modelG2,modelG3,modelD1,modelD2,modelD3,trainDataloader1, trainDataloader2, trainDataloader3,window_size1,window_size2,window_size3,y_scaler,train_x1,
                         train_x2, train_x3, train_y,val_x1,val_x2,val_x3,val_y,distill,num_epochs,
                     output_dir,device):

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5

    # 二元交叉熵【损失函数，可能会有问题
    criterion = nn.BCELoss()

    optimizerG1 = torch.optim.AdamW(modelG1.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
    optimizerG2 = torch.optim.AdamW(modelG2.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
    optimizerG3 = torch.optim.AdamW(modelG3.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))


    # 为每个优化器设置 ReduceLROnPlateau 调度器
    schedulerG1 = lr_scheduler.ReduceLROnPlateau(optimizerG1, mode='min', factor=0.1, patience=16, min_lr=1e-7)
    schedulerG2 = lr_scheduler.ReduceLROnPlateau(optimizerG2, mode='min', factor=0.1, patience=16, min_lr=1e-7)
    schedulerG3 = lr_scheduler.ReduceLROnPlateau(optimizerG3, mode='min', factor=0.1, patience=16, min_lr=1e-7)

    optimizerD1 = torch.optim.Adam(modelD1.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
    optimizerD2 = torch.optim.Adam(modelD2.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
    optimizerD3 = torch.optim.Adam(modelD3.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))

    generators=[modelG1,modelG2,modelG3]
    dataloaders=[trainDataloader1,trainDataloader2,trainDataloader3]
    window_sizes=[window_size1,window_size2,window_size3]
    optimizers=[optimizerG1,optimizerG2,optimizerG3]

    # 参数配置，需要反复调整，从而获得比较理想的实验结果
    alpha1, alpha2, alpha3,alpha4= 0.8, 0.1,0.1, 1.0
    beta1, beta2, beta3 ,beta4= 0.1, 0.8,0.1, 1.0
    gamma1, gamma2, gamma3,gamma4= 0.1, 0.1,0.8, 1.0

    best_epoch1 = best_epoch2 = best_epoch3 = -1  #

    # 用于记录损失值
    histG1 = np.zeros(num_epochs)
    histG2 = np.zeros(num_epochs)
    histG3 = np.zeros(num_epochs)  # 添加G3的损失记录

    histD1 = np.zeros(num_epochs)
    histD2 = np.zeros(num_epochs)
    histD3 = np.zeros(num_epochs)  # 添加D3的损失记录

    # 用于记录两个判别器对于两个生成器的单独的损失
    histD1_G1 = np.zeros(num_epochs)
    histD2_G1 = np.zeros(num_epochs)
    histD3_G1 = np.zeros(num_epochs)

    histD1_G2 = np.zeros(num_epochs)
    histD2_G2 = np.zeros(num_epochs)
    histD3_G2 = np.zeros(num_epochs)

    histD1_G3 = np.zeros(num_epochs)
    histD2_G3 = np.zeros(num_epochs)
    histD3_G3 = np.zeros(num_epochs)

    # 主要是用于记录MSE的变化，看一下变化曲线
    hist_MSE_G1 = np.zeros(num_epochs)
    hist_MSE_G2 = np.zeros(num_epochs)
    hist_MSE_G3 = np.zeros(num_epochs)

    # 用来记录训练过程中的验证集上的损失和分数
    hist_val_loss1 = np.zeros(num_epochs)
    hist_val_loss2 = np.zeros(num_epochs)
    hist_val_loss3 = np.zeros(num_epochs)

    best_mse1 = float('inf')
    best_mse2 = float('inf')
    best_mse3 = float('inf')

    best_model_state1 = None
    best_model_state2 = None
    best_model_state3 = None

    patience_counter = 0
    patience = 50
    feature_num = train_x1.shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    for epoch in range(num_epochs):

        modelG1.train()
        modelG2.train()
        modelG3.train()

        lossdata_G1 = []
        lossdata_G2 = []
        lossdata_G3 = []

        lossdata_D1 = []
        lossdata_D2 = []
        lossdata_D3 = []

        lossdata_D1_G1 = []
        lossdata_D2_G1 = []
        lossdata_D3_G1 = []

        lossdata_D1_G2 = []
        lossdata_D2_G2 = []
        lossdata_D3_G2 = []

        lossdata_D1_G3 = []
        lossdata_D2_G3 = []
        lossdata_D3_G3 = []

        lossdata_MSE_G1 = []
        lossdata_MSE_G2 = []
        lossdata_MSE_G3 = []


        gap13=window_size3-window_size1
        gap23=window_size3-window_size2
        for batch_idx, (x3,y3)in enumerate(trainDataloader3):

            x3 = x3.to(device)
            y3 = y3.to(device)
            x1 =x3[:,gap13:,:]
            y1 = y3[:,gap13:,:]
            x2 = x3[:,gap23:,:]
            y2 = y3[:,gap23:,:]

            modelG1.eval()
            modelG2.eval()
            modelG3.eval()
            modelD1.train()
            modelD2.train()
            modelD3.train()

            #discriminator output for real data
            dis_real_output1 = modelD1(y1)
            dis_real_output2 = modelD2(y2)
            dis_real_output3 = modelD3(y3)

            real_labels_1 = torch.ones_like(dis_real_output1).to(device)
            real_labels_2 = torch.ones_like(dis_real_output2).to(device)
            real_labels_3 = torch.ones_like(dis_real_output3).to(device)

            # 判别器对真实数据损失
            lossD_real1 = criterion(dis_real_output1, real_labels_1)
            lossD_real2 = criterion(dis_real_output2, real_labels_2)
            lossD_real3 = criterion(dis_real_output3, real_labels_3)

            # discriminator output for fake data
            # G1生成的数据
            fake_data_temp_G1 = modelG1(x1).detach()
            # G2生成的数据
            fake_data_temp_G2 = modelG2(x2).detach()
            # G3生成的数据
            fake_data_temp_G3 = modelG3(x3).detach()


            # 拼接之后可以让生成的假数据，既包含假数据又包含真数据，
            fake_data_G1 = torch.cat([y1[:, :window_size1, :], fake_data_temp_G1.reshape(-1, 1, target_num)], axis=1)
            fake_data_G2 = torch.cat([y2[:, :window_size2, :], fake_data_temp_G2.reshape(-1, 1, target_num)], axis=1)
            fake_data_G3 = torch.cat([y3[:, :window_size3, :], fake_data_temp_G3.reshape(-1, 1, target_num)], axis=1)


            #判别器对伪造数据损失
            # 三个生成器的结果的数据对齐
            fake_data_1to2 = torch.cat([y2[:, :window_size2-window_size1, :], fake_data_G1], axis=1)
            fake_data_1to3 = torch.cat([y3[:, :window_size3-window_size1, :], fake_data_G1], axis=1)

            fake_data_2to1 = fake_data_G2[:, window_size2 - window_size1:, :]
            fake_data_2to3 = torch.cat([y3[:, :window_size3 - window_size2, :], fake_data_G2], axis=1)

            fake_data_3to1 = fake_data_G3[:, window_size3 - window_size1:, :]
            fake_data_3to2 = fake_data_G3[:, window_size3 - window_size2:, :]

            # 三个判别器，对于三个生成器伪造结果的判断
            dis_fake_outputD1_1 = modelD1(fake_data_G1)
            dis_fake_outputD1_2 = modelD1(fake_data_2to1)
            dis_fake_outputD1_3 = modelD1(fake_data_3to1)

            dis_fake_outputD2_1 = modelD2(fake_data_1to2)
            dis_fake_outputD2_2 = modelD2(fake_data_G2)
            dis_fake_outputD2_3 = modelD2(fake_data_3to2)

            dis_fake_outputD3_1 = modelD3(fake_data_1to3)
            dis_fake_outputD3_2 = modelD3(fake_data_2to3)
            dis_fake_outputD3_3 = modelD3(fake_data_G3)

            fake_labels_1 = torch.zeros_like(real_labels_1).to(device)
            fake_labels_2 = torch.zeros_like(real_labels_2).to(device)
            fake_labels_3 = torch.zeros_like(real_labels_3).to(device)


            # 三个判别器对伪造数据损失
            lossD1_fake1 = criterion(dis_fake_outputD1_1, fake_labels_1)
            lossD1_fake2 = criterion(dis_fake_outputD1_2, fake_labels_1)
            lossD1_fake3 = criterion(dis_fake_outputD1_3, fake_labels_1)

            lossD2_fake1 = criterion(dis_fake_outputD2_1, fake_labels_2)
            lossD2_fake2 = criterion(dis_fake_outputD2_2, fake_labels_2)
            lossD2_fake3 = criterion(dis_fake_outputD2_3, fake_labels_2)

            lossD3_fake1 = criterion(dis_fake_outputD3_1, fake_labels_3)
            lossD3_fake2 = criterion(dis_fake_outputD3_2, fake_labels_3)
            lossD3_fake3 = criterion(dis_fake_outputD3_3, fake_labels_3)

            # 两个判别器的总的损失和
            loss_D1 = alpha1 * lossD1_fake1 + alpha2 * lossD1_fake2 +alpha3*lossD1_fake3 +alpha4*lossD_real1

            loss_D2 = beta1 * lossD2_fake1 + beta2 * lossD2_fake2 + beta3 * lossD2_fake3+ beta4*lossD_real2

            loss_D3 = gamma1 * lossD3_fake1 + gamma2 * lossD3_fake2 + gamma3 * lossD3_fake3 + gamma4 * lossD_real3

            lossdata_D1.append(loss_D1.item())
            lossdata_D2.append(loss_D2.item())
            lossdata_D3.append(loss_D3.item())

            lossdata_D1_G1.append(lossD1_fake1.item())
            lossdata_D2_G1.append(lossD2_fake1.item())
            lossdata_D3_G1.append(lossD3_fake1.item())
            lossdata_D1_G2.append(lossD1_fake2.item())
            lossdata_D2_G2.append(lossD2_fake2.item())
            lossdata_D3_G2.append(lossD3_fake2.item())
            lossdata_D1_G3.append(lossD1_fake3.item())
            lossdata_D2_G3.append(lossD2_fake3.item())
            lossdata_D3_G3.append(lossD3_fake3.item())

            # 根据批次的奇偶性交叉训练两个GAN
            #if batch_idx% 2 == 0:
            optimizerD1.zero_grad()
            optimizerD2.zero_grad()
            optimizerD3.zero_grad()
            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()
            optimizerD1.step()
            optimizerD2.step()
            optimizerD3.step()

            '''训练生成器'''
            modelD1.eval()
            modelD2.eval()
            modelD3.eval()
            modelG1.train()
            modelG2.train()
            modelG3.train()

            fake_data_temp_G1 = modelG1(x1)
            fake_data_temp_G2 = modelG2(x2)
            fake_data_temp_G3 = modelG3(x3)

            fake_data_G1 = torch.cat([y1[:, :window_size1, :], fake_data_temp_G1.reshape(-1, 1, target_num)], axis=1)
            fake_data_G2 = torch.cat([y2[:, :window_size2, :], fake_data_temp_G2.reshape(-1, 1, target_num)], axis=1)
            fake_data_G3 = torch.cat([y3[:, :window_size3, :], fake_data_temp_G3.reshape(-1, 1, target_num)], axis=1)

            fake_data_1to2 = torch.cat([y2[:, :window_size2 - window_size1, :], fake_data_G1], axis=1)
            fake_data_1to3 = torch.cat([y3[:, :window_size3 - window_size1, :], fake_data_G1], axis=1)

            fake_data_2to1 = fake_data_G2[:, window_size2 - window_size1:, :]
            fake_data_2to3 = torch.cat([y3[:, :window_size3 - window_size2, :], fake_data_G2], axis=1)

            fake_data_3to1 = fake_data_G3[:, window_size3 - window_size1:, :]
            fake_data_3to2 = fake_data_G3[:, window_size3 - window_size2:, :]

            dis_fake_outputD1_1 = modelD1(fake_data_G1)
            dis_fake_outputD1_2 = modelD1(fake_data_2to1)
            dis_fake_outputD1_3 = modelD1(fake_data_3to1)

            dis_fake_outputD2_1 = modelD2(fake_data_1to2)
            dis_fake_outputD2_2 = modelD2(fake_data_G2)
            dis_fake_outputD2_3 = modelD2(fake_data_3to2)

            dis_fake_outputD3_1 = modelD3(fake_data_1to3)
            dis_fake_outputD3_2 = modelD3(fake_data_2to3)
            dis_fake_outputD3_3 = modelD3(fake_data_G3)

            lossG1_D1 = criterion(dis_fake_outputD1_1, real_labels_1)
            lossG1_D2 = criterion(dis_fake_outputD2_1, real_labels_2)
            lossG1_D3 = criterion(dis_fake_outputD3_1, real_labels_3)

            lossG2_D1 = criterion(dis_fake_outputD1_2, real_labels_1)
            lossG2_D2 = criterion(dis_fake_outputD2_2, real_labels_2)
            lossG2_D3 = criterion(dis_fake_outputD3_2, real_labels_3)

            lossG3_D1 = criterion(dis_fake_outputD1_3, real_labels_1)
            lossG3_D2 = criterion(dis_fake_outputD2_3, real_labels_2)
            lossG3_D3 = criterion(dis_fake_outputD3_3, real_labels_3)

            loss_G1 = alpha1 * lossG1_D1 + alpha2 * lossG1_D2 + alpha3 * lossG1_D3
            loss_G2 = beta1 * lossG2_D1 + beta2 * lossG2_D2 + beta3 * lossG2_D3
            loss_G3 = gamma1 * lossG3_D1 + gamma2 * lossG3_D2 + gamma3 * lossG3_D3

            loss_mse_G1 = F.mse_loss(fake_data_temp_G1.squeeze(), y1[:, -1, :].squeeze())
            loss_G1 += loss_mse_G1

            loss_mse_G2 = F.mse_loss(fake_data_temp_G2.squeeze(), y2[:, -1, :].squeeze())
            loss_G2 += loss_mse_G2

            loss_mse_G3 = F.mse_loss(fake_data_temp_G3.squeeze(), y3[:, -1, :].squeeze())
            loss_G3 += loss_mse_G3

            lossdata_MSE_G1.append(loss_mse_G1.item())
            lossdata_MSE_G2.append(loss_mse_G2.item())
            lossdata_MSE_G3.append(loss_mse_G3.item())

            lossdata_G1.append(loss_G1.item())
            lossdata_G2.append(loss_G2.item())
            lossdata_G3.append(loss_G3.item())

            optimizerG1.zero_grad()
            optimizerG2.zero_grad()
            optimizerG3.zero_grad()
            loss_G1.backward()
            loss_G2.backward()
            loss_G3.backward()
            optimizerG1.step()
            optimizerG2.step()
            optimizerG3.step()

        histG1[epoch] = np.mean(lossdata_G1)
        histG2[epoch] = np.mean(lossdata_G2)
        histG3[epoch] = np.mean(lossdata_G3)

        histD1[epoch] = np.mean(lossdata_D1)
        histD2[epoch] = np.mean(lossdata_D2)
        histD3[epoch] = np.mean(lossdata_D3)

        histD1_G1[epoch] = np.mean(lossdata_D1_G1)
        histD2_G1[epoch] = np.mean(lossdata_D2_G1)
        histD3_G1[epoch] = np.mean(lossdata_D3_G1)

        histD1_G2[epoch] = np.mean(lossdata_D1_G2)
        histD2_G2[epoch] = np.mean(lossdata_D2_G2)
        histD3_G2[epoch] = np.mean(lossdata_D3_G2)

        histD1_G3[epoch] = np.mean(lossdata_D1_G3)
        histD2_G3[epoch] = np.mean(lossdata_D2_G3)
        histD3_G3[epoch] = np.mean(lossdata_D3_G3)

        hist_MSE_G1[epoch] = np.mean(lossdata_MSE_G1)
        hist_MSE_G2[epoch] = np.mean(lossdata_MSE_G2)
        hist_MSE_G3[epoch] = np.mean(lossdata_MSE_G3)

        # 使用验证集对Generator的生成效果进行验证

        hist_val_loss1[epoch] = validate(modelG1, val_x1, val_y)
        hist_val_loss2[epoch] = validate(modelG2, val_x2, val_y)
        hist_val_loss3[epoch] = validate(modelG3, val_x3, val_y)

        improved = [False] * 3
        if hist_val_loss1[epoch] < best_mse1:
            best_mse1 = hist_val_loss1[epoch]
            best_model_state1 = copy.deepcopy(modelG1.state_dict())
            best_epoch1 = epoch + 1
            improved[0] = True

        if hist_val_loss2[epoch] < best_mse2:
            best_mse2 = hist_val_loss2[epoch]
            best_model_state2 = copy.deepcopy(modelG2.state_dict())
            best_epoch2 = epoch + 1
            improved[1] = True

        if hist_val_loss3[epoch] < best_mse3:
            best_mse3= hist_val_loss3[epoch]
            best_model_state3 = copy.deepcopy(modelG3.state_dict())
            best_epoch3 = epoch + 1
            improved[2] = True

        # 假设val_loss是验证集的损失
        schedulerG1.step(hist_val_loss1[epoch])
        schedulerG2.step(hist_val_loss2[epoch])
        schedulerG3.step(hist_val_loss3[epoch])

        if distill and epoch>10:
            losses = [hist_val_loss1[epoch], hist_val_loss2[epoch], hist_val_loss3[epoch]]
            rank = np.argsort(losses)
            do_distill(rank, generators, dataloaders, optimizers, window_sizes,device)

        # 每个epoch结束时，打印训练过程中的损失
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Validation MSE G1: {hist_val_loss1[epoch]:.8f}, G2: {hist_val_loss2[epoch]:.8f}, G3: {hist_val_loss3[epoch]:.8f}")
        print(f"patience counter:{patience_counter}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    #visualize generator loss
    data_G1 = [histD1_G1, histD2_G1, histD3_G1, histG1]
    data_G2 = [histD1_G2, histD2_G2, histD3_G2, histG2]
    data_G3 = [histD1_G3, histD2_G3, histD3_G3, histG3]
    plot_generator_losses(data_G1, data_G2, data_G3,output_dir)
    #visualize discriminator loss
    data_D1 = [histD1_G1, histD1_G2, histD1_G3, histG1]
    data_D2 = [histD2_G1, histD2_G2, histD2_G3, histG2]
    data_D3 = [histD3_G1, histD3_G2, histD3_G3, histG3]
    plot_discriminator_losses(data_D1, data_D2, data_D3,output_dir)
    #overall G&D
    visualize_overall_loss(histG1, histG2, histG3,histD1, histD2, histD3,output_dir)
    plot_mse_loss(hist_MSE_G1[:epoch], hist_MSE_G2[:epoch], hist_MSE_G3[:epoch], hist_val_loss1[:epoch], hist_val_loss2[:epoch], hist_val_loss3[:epoch], epoch,output_dir)

    print("G1 best epoch:",best_epoch1)
    print("G2 best epoch:",best_epoch2)
    print("G3 best epoch:",best_epoch3)


    results =evaluate_best_models(modelG1, modelG2, modelG3, best_model_state1, best_model_state2, best_model_state3, train_x1,
                         train_x2, train_x3, train_y, val_x1, val_x2, val_x3, val_y,y_scaler, output_dir)
    return results



def do_distill(rank, generators, dataloaders,optimizers,window_sizes,device):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    #term of teacher is longer
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y) in enumerate(distill_dataloader):

        y=y[:,-1,:]
        y=y.to(device)
        if gap>0:
            x_teacher=x
            x_student=x[:,gap:,:]
        else:
            x_teacher=x[:,(-1)*gap:,:]
            x_student=x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output = teacher_generator(x_teacher).detach()

        # Forward pass with student generator
        student_output = student_generator(x_student)

        # Calculate distillation loss (MSE between teacher and student generator's outputs)
        soft_loss = F.mse_loss(student_output, teacher_output)
        hard_loss = F.mse_loss(student_output, y)
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        distillation_loss.backward()
        student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed


def refine_best_models_with_real_data_v2(
    G_rank, D_rank, generators, discriminators, g_optimizers, d_optimizers,
    dataloaders, window_sizes, device_G="cuda:0", device_D="cuda:1"
):
    best_G_idx = G_rank[0]
    best_D_idx = D_rank[0]

    generator = generators[best_G_idx]
    discriminator = discriminators[best_D_idx]
    g_optimizer = g_optimizers[best_G_idx]
    d_optimizer = d_optimizers[best_D_idx]
    dataloader_G = dataloaders[best_G_idx]
    dataloader_D = dataloaders[best_D_idx]
    window_size_D = window_sizes[best_D_idx]

    def train_generator():
        generator.to(device_G)
        generator.train()
        for x, y in dataloader_G:
            x = x.to(device_G)
            y = y[:, -1, :].to(device_G)  # 用最后一个时间步
            g_optimizer.zero_grad()
            pred = generator(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            g_optimizer.step()

    def train_discriminator():
        discriminator.to(device_D)
        discriminator.train()
        for _, y in dataloader_D:
            y_real = y[:, -window_size_D:, :].to(device_D)
            y_fake = y_real + torch.randn_like(y_real) * 0.05
            label_real = torch.ones((y_real.size(0), 1)).to(device_D)
            label_fake = torch.zeros((y_fake.size(0), 1)).to(device_D)

            d_optimizer.zero_grad()
            out_real = discriminator(y_real)
            out_fake = discriminator(y_fake)
            loss_real = F.binary_cross_entropy(out_real, label_real)
            loss_fake = F.binary_cross_entropy(out_fake, label_fake)
            loss = loss_real + loss_fake
            loss.backward()
            d_optimizer.step()

    # # 使用 torch.multiprocessing 并行跑 G 和 D
    # import torch.multiprocessing as mp
    # p1 = mp.Process(target=train_generator)
    # p2 = mp.Process(target=train_discriminator)
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()


