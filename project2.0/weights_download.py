try:
    import urllib.request
except:
    print("****** urllib package not installed - cannot fetch the solutions.******")
try:
    # Resulting weights for section "Fine tunning"
    print("Downloading weight files. This may take a few minutes.")
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1oVNCEzk3Ml_HTrN5CxlCOFZcZUrhwKhQ&authuser=0&export=download",
        'weights_no_sim_img_False_t_2_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1IxOnr5Q81uBHgxuJacVO5PdU0Eo2zNVp&authuser=0&export=download",
        'weights_no_sim_img_False_t_2_rows.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1SqRAN_Jo1h-WsZmyoJZobtdZxXhzUZP3&authuser=0&export=download",
        'weights_no_sim_img_False_t_4_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1Ig50xB6z7ceQbqM5rJQGwZV6U3cdvuyL&authuser=0&export=download",
        'weights_no_sim_img_False_t_4_rows.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=10tKi3Js4XVt_wBIz-EAtAqR2jkddOiWr&authuser=0&export=download",
        'weights_no_sim_img_False_t_5_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1CzPRLsaFpYwc-YkUzQTGJ3kom4RXpG7S&authuser=0&export=download",
        'weights_no_sim_img_False_t_5_rows.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1-Vdbo7QXWVkXy4UsTkoErDGaDoIaiCKw&authuser=0&export=download",
        'is_img_or_doc.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1IVgv0Q9ZfIEd2L7PzXAPFOc39RnzKNxQ&authuser=0&export=download",
        'weights_no_sim_img_True_t_2_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1D7d5BwpmMtNrocGZiAMHbfwy_ik2Bbma&authuser=0&export=download",
        'weights_no_sim_img_True_t_2_rows.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1bJuj6tPSbwpFsOLhfYsHh0M7LNU8X3y0&authuser=0&export=download",
        'weights_no_sim_img_True_t_4_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1Z3_h_Qlmx4WvMvqI4c3kF5ym6ZJZdoSW&authuser=0&export=download",
        'weights_no_sim_img_True_t_4_rows.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1QWg7C2iu3NzktceNkW7eYp7t55q1B-Qw&authuser=0&export=download",
        'weights_no_sim_img_True_t_5_cols.h5')
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?id=1ryixLjM4_hBvThrzDhHqhFW3-pZ9ZElq&authuser=0&export=download",
        'weights_no_sim_img_True_t_5_rows.h5')

    print("Completed weight downloads.")
except:
    print("****** \nCannot auto-download the solution weights.\nPlease use the following link to download manually: "
          "https://drive.google.com/open?id=1aYCefWtPdV06L7dlHxDsdC_jRfWgf3j0 \n******")