try:
    import urllib.request
except:
    print("****** urllib package not installed - cannot fetch the solutions.******")
try:
    # Resulting weights for section "Fine tunning"
    print("Downloading our solution weight files. This may take a few minutes (315 MB total).")
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1zr16MMaMdYe06D_YiPqkSVCcCCU93MS1&authuser=0&export=download",
        'resnet_maxSize_32_t_5_isImg_False.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1xnJUYvlCmi86BGMcgdSC_Bhi-0vwq2ht&authuser=0&export=download",
        'resnet_maxSize_32_t_5_isImg_True.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=17ND6soRS86zmct1SxY8UwxO2vSm9D2FL&authuser=0&export=download",
        'resnet_maxSize_32_t_4_isImg_False.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1fk--OsWqIp9JjLwoBcme0RgwrEAuhhnA&authuser=0&export=download",
        'resnet_maxSize_32_t_4_isImg_True.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=12fcuqkor0coPUdc5xmlE-J7NXNCJ4lzV&authuser=0&export=download",
        'resnet_maxSize_32_t_2_isImg_False.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1QJuc_FLmjshPcJMiNLjs_ygrHoOr3VlD&authuser=0&export=download",
        'resnet_maxSize_32_t_2_isImg_True.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1-Vdbo7QXWVkXy4UsTkoErDGaDoIaiCKw&authuser=0&export=download",
        'is_img_or_doc.h5')
    print("Completed weight downloads.")
except:
    print("****** \nCannot auto-download the solution weights.\nPlease use the following link to download manually: "
          "https://drive.google.com/open?id=1aYCefWtPdV06L7dlHxDsdC_jRfWgf3j0 \n******")