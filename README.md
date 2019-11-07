# DCGAN_check_epoch_better
follow dcgan tutorial and try to be better

Brief introduction
    Using this script you can generate human face, simultaneously check which epoch can generate better resolution.

In this code, you can change the following code to decide how many image will generate in an epoch. i is the batch size. There is 1583 batches, so I set 1552 will generate 30 images in each epoch.
    if(epoch >= 0) and (i > 1552):  

            # Create batch of latent vectors that we will use to visualize
            #  the progression of the generator
            fixed_noise = torch.randn(64, nz, 1, 1, device=device)         
            #img_list.append(vutils.make_grid(fake[-9:], nrow=3, normalize=True).permute(0, 2, 3 ,1).numpy())
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            #print(type(fake))
            img_list.append(fake[-9:].permute(0, 2, 3 ,1).numpy())
            #print(type(fake[-9:].permute(0, 2, 3 ,1).numpy()))
