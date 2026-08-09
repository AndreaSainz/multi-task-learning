
    
def inverse_gradient_only_pipeline(y, x, A_torch, AT_torch, alpha, gamma, R_ini, R_min, R_max, I, decay_iters, H, W, x_true, centre_true, digit_true, regularizer, batch_id, results_dir, name, accelerator, verbose_images, verbose_metrics, verbose_convergence, verbose_gradient, **parameters):
    
    regularizer = regularizer.strip().lower()


    # initilizing pipeline variables
    images_SIRT = []
    images = []
    psnr_history = []
    ssim_history = []   

    if regularizer == "bayes":
        images_class = []
        images_deno = []
    
    
    # extracting batch size
    B = y.shape[0]


    ############################################ Initilizing tasks ############################################


    # RECONSTRUCTION
    x = x.clone()
    x_sirt = torch.zeros(B, 1, H, W, device=y.device)

    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        dx_norm_mean = torch.zeros(R_ini, device=y.device) 
        dx_norm_b0 = torch.zeros(R_ini, device=y.device)  


    for r in range(R_ini):
        x_prev = x_sirt.clone()
        x_sirt , step_data = sirt_step(A_torch, AT_torch, x_sirt, y, alpha)

        # looking into the gradient size
        if verbose_gradient and accelerator.is_main_process and batch_id ==0 and r % 10 == 0:
            logger.info(f"Gradient descent step -1_{r}: {step_data.flatten(1).norm(dim=1).mean().item()}")

        # debugging || x^{k+1} -x^k|| --> convergence 
        if verbose_convergence and accelerator.is_main_process and batch_id == 0:
            dx = x_sirt - x_prev 
            dx_norm_b0[r]   = dx[0].flatten().norm()
            dx_norm_mean[r] = dx.flatten(1).norm(dim=1).mean().item()

    
    #plotting the rate convergence
    if verbose_convergence and accelerator.is_main_process and batch_id == 0:
        plt.figure() 
        plt.plot(dx_norm_mean.detach().cpu().numpy(), label="mean over batch")
        plt.plot(dx_norm_b0.detach().cpu().numpy(), label="sample 0")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel("$||x_{t+1}-x_t||$")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{batch_id}_initial_dx_norm_batch0_{name}.png"), dpi=300, bbox_inches="tight")
        plt.close()


    # Metrics after the initialization with SIRT (equi to I = -1, with R_ini steps)
    crop_x_true = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
    crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)

    for b in range(B):
        cy = int(centre_true[b, 0].item())
        cx = int(centre_true[b, 1].item())
        crop_x_true[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        crop_x[b:b+1] = x[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
        crop_x_sirt[b:b+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]

        
    psnr_x_init = compute_psnr_results(x.detach(), x_true.detach())
    psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())

    ssim_x_init = compute_ssim(x.detach(), x_true.detach(), data_range=1.0)
    ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
    
    psnr_x_init_crop = compute_psnr_results(crop_x.detach(), crop_x_true.detach())
    psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x_true.detach())
    
    ssim_x_init_crop = compute_ssim(crop_x.detach(), crop_x_true.detach(), data_range=1.0)
    ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x_true.detach(), data_range=1.0)


    ssim_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "ssim_x": ssim_x_init.detach().cpu(),
    "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
    "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
    "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        ssim_entry["ssim_x_clas"] = ssim_sirt_init.detach().cpu()
        ssim_entry["ssim_x_deno"] = ssim_sirt_init.detach().cpu()
        ssim_entry["ssim_x_clas_crop"] = ssim_sirt_init_crop.detach().cpu()
        ssim_entry["ssim_x_deno_crop"] = ssim_sirt_init_crop.detach().cpu()

    ssim_history.append(ssim_entry)


    if verbose_metrics and accelerator.is_main_process:
        logger.info(
            f"[PSNR] I=-1 | steps={R_ini} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
        )
        logger.info(
            f"[SSIM] I=-1 | steps={R_ini} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
        )


    psnr_entry = {
    "I": -1,
    "num_inner_steps": R_ini,
    "psnr_x": psnr_x_init.detach().cpu(),
    "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
    "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
    "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
    }

    if regularizer == "bayes":
        psnr_entry["psnr_x_clas"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_deno"] = psnr_x_init.detach().cpu()
        psnr_entry["psnr_x_clas_crop"] = psnr_x_init_crop.detach().cpu()
        psnr_entry["psnr_x_deno_crop"] = psnr_x_init_crop.detach().cpu()

    psnr_history.append(psnr_entry)

    


    # copying x for all the cases (based in regularizer)
    x = x.clamp(0,1)
    x_both = x.clone()


    if regularizer == "bayes":
        x_class = x.clone()
        x_deno = x.clone()



    ###### DIGIT 
    # taking the true digit 
    digit = torch.nn.functional.one_hot(digit_true, num_classes=10).float()
    digit = digit.to(x.device)
    digit_true = digit_true.to(x.device)

    ####### CENTRE 
    # taking the true centre 
    centre_true = centre_true.to(x.device)
    centre = centre_true.clone()
    

    

    images.append(x.detach().cpu())
    images_SIRT.append(x_sirt.detach().cpu())
    if regularizer == "bayes":
        images_class.append(x_class.detach().cpu())
        images_deno.append(x_deno.detach().cpu())
    

    ####################################### Conditional Loop  #######################################

    for i in range(I):
        if i < decay_iters:
            if decay_iters > 1:
                R = int(R_max - (R_max - R_min) * i / (decay_iters - 1))
            else:
                R = R_min
        else:
            R = R_min
        
        # reconstruction step using only the gradient
        for r in range(R):

            # BOTH image, meaning the one using the complete gradient
            x_both , _ = update_case(x_both, y, A_torch, AT_torch, centre, digit, alpha, gamma, "both", i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            # SIRT image
            x_sirt , _ = update_case(x_sirt, y, A_torch, AT_torch, centre, digit, alpha, gamma, "none", i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

            if regularizer == "bayes":
                #only classifier
                x_class , _ = update_case(x_class, y, A_torch, AT_torch, centre, digit, alpha, gamma, "class", i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)

                #only denoiser
                x_deno , _ = update_case(x_deno, y, A_torch, AT_torch, centre, digit, alpha, gamma, "deno", i, r, batch_id, regularizer, results_dir, name, verbose_gradient, **parameters)


        # taking crops
        crop_x = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_both = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
        crop_x_sirt = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)

        if regularizer == "bayes":
            crop_x_class = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)
            crop_x_deno = torch.zeros(B, 1, 28, 28, device=y.device, dtype=x.dtype)

        for b in range(B):
            cy = int(centre_true[b, 0].item())
            cx = int(centre_true[b, 1].item())
            crop_x[b:b+1] = x_true[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_both[b:b+1] = x_both[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            crop_x_sirt[b:b+1] = x_sirt[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
            if regularizer == "bayes":
                crop_x_class[b:b+1] = x_class[b:b+1, :, cy-14:cy+14, cx-14:cx+14]
                crop_x_deno[b:b+1] = x_deno[b:b+1, :, cy-14:cy+14, cx-14:cx+14]


        # computing metrics 
        psnr_x_init = compute_psnr_results(x_both.detach(), x_true.detach())
        psnr_sirt_init = compute_psnr_results(x_sirt.detach(), x_true.detach())

        ssim_x_init = compute_ssim(x_both.detach(), x_true.detach(), data_range=1.0)
        ssim_sirt_init = compute_ssim(x_sirt.detach(), x_true.detach(), data_range=1.0)
        
        psnr_x_init_crop = compute_psnr_results(crop_x_both.detach(), crop_x.detach())
        psnr_sirt_init_crop = compute_psnr_results(crop_x_sirt.detach(), crop_x.detach())
        
        ssim_x_init_crop = compute_ssim(crop_x_both.detach(), crop_x.detach(), data_range=1.0)
        ssim_sirt_init_crop = compute_ssim(crop_x_sirt.detach(), crop_x.detach(), data_range=1.0)
        

        if regularizer == "bayes":
            psnr_x_clas = compute_psnr_results(x_class.detach(), x_true.detach())
            psnr_x_deno = compute_psnr_results(x_deno.detach(), x_true.detach())
            ssim_x_clas = compute_ssim(x_class.detach(), x_true.detach(), data_range=1.0)
            ssim_x_deno = compute_ssim(x_deno.detach(), x_true.detach(), data_range=1.0)
            psnr_x_clas_crop = compute_psnr_results(crop_x_class.detach(), crop_x.detach())
            psnr_x_deno_crop = compute_psnr_results(crop_x_deno.detach(), crop_x.detach())
            ssim_x_clas_crop = compute_ssim(crop_x_class.detach(), crop_x.detach(), data_range=1.0)
            ssim_x_deno_crop = compute_ssim(crop_x_deno.detach(), crop_x.detach(), data_range=1.0)


        #saving metrics
        ssim_entry = {
        "I": i,
        "num_inner_steps": R,
        "ssim_x": ssim_x_init.detach().cpu(),
        "ssim_x_sirt": ssim_sirt_init.detach().cpu(),
        "ssim_x_crop": ssim_x_init_crop.detach().cpu(),
        "ssim_x_sirt_crop": ssim_sirt_init_crop.detach().cpu()
        }

        if regularizer == "bayes":
            ssim_entry["ssim_x_clas"] = ssim_x_clas.detach().cpu()
            ssim_entry["ssim_x_deno"] = ssim_x_deno.detach().cpu()
            ssim_entry["ssim_x_clas_crop"] = ssim_x_clas_crop.detach().cpu()
            ssim_entry["ssim_x_deno_crop"] = ssim_x_deno_crop.detach().cpu()

        ssim_history.append(ssim_entry)
        

        psnr_entry = {
        "I": i,
        "num_inner_steps": R,
        "psnr_x": psnr_x_init.detach().cpu(),
        "psnr_x_sirt": psnr_sirt_init.detach().cpu(),
        "psnr_x_crop": psnr_x_init_crop.detach().cpu(),
        "psnr_x_sirt_crop": psnr_sirt_init_crop.detach().cpu(),
        }

        if regularizer == "bayes":
            psnr_entry["psnr_x_clas"] = psnr_x_clas.detach().cpu()
            psnr_entry["psnr_x_deno"] = psnr_x_deno.detach().cpu()
            psnr_entry["psnr_x_clas_crop"] = psnr_x_clas_crop.detach().cpu()
            psnr_entry["psnr_x_deno_crop"] = psnr_x_deno_crop.detach().cpu()

        psnr_history.append(psnr_entry)

    

        if verbose_metrics and batch_id == 0 and accelerator.is_main_process:
            logger.info(
                f"[PSNR] I={i} | steps={R} | x={psnr_x_init.mean().item():.4f} dB | x_sirt={psnr_sirt_init.mean().item():.4f} dB"
            )
            logger.info(
                f"[SSIM] I={i} | steps={R} | x={ssim_x_init.mean().item():.4f} | x_sirt={ssim_sirt_init.mean().item():.4f}"
            )


       
        if verbose_images and accelerator.is_main_process and batch_id == 0 and i % 2 == 0: 
            for b in range(B):

                # --- SIRT ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_sirt[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_sirt_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

                # --- BOTH ---
                fig, ax = plt.subplots()
                im = ax.imshow(x_both[b,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax)
                fig.savefig(
                    os.path.join(results_dir, f"{batch_id}_{i}_sample{b}_flow_{name}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close(fig)

        

        

        #save information
        images.append(x_both.detach().cpu())
        images_SIRT.append(x_sirt.detach().cpu())

        if regularizer == "bayes":
            images_class.append(x_class.detach().cpu())
            images_deno.append(x_deno.detach().cpu())
        

            del crop_x_class, crop_x_deno

        del crop_x, crop_x_both, crop_x_sirt
        torch.cuda.empty_cache()

    if regularizer == "bayes":
        return images, images_SIRT, images_class, images_deno, psnr_history, ssim_history
    
    return images, images_SIRT, psnr_history, ssim_history