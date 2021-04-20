## FACE DETECTING
**1. Tổng quan**

- bài toán: Đưa ra quyết định một bức ảnh có phải làm một khuôn mặt hay không.
- cách tiếp cận: Sử dụng thuật toán học máy Gaussian Naive Bayes để phân lớp các bức ảnh thành 2 nhãn: là và không là khuôn mặt.
- dữ liệu về khuôn mặt: 200000 ảnh khuôn mặt và 10000 bức ảnh tự trích xuất.

**2. Giải quyết vấn đề**

a. Trích chọn đặc trưng

- Đưa kích thước ảnh về một kích cỡ cho trước (80 x 80)
- Mỗi bức ảnh được chuyển về dạng đen và trắng. Sau bước này, bức ảnh là một ma trận 2 chiều, trong đó mỗi biễu diễn giá trị thuộc khoảng [0, 255] thể hiện "độ sáng" của điểm ảnh.
![gray transfer](https://www.google.com/url?sa=i&url=https://cppsecrets.com/users/204211510411798104971091085153504964103109971051084699111109/C00-OpenCV-program-to-convert-BGR-image-to-grayscale-image.php&psig=AOvVaw2Kh5dvj-1LCBeIkJbPaU4H&ust=1618971275822000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKCSlJPgi_ACFQAAAAAdAAAAABAD)
- Tách các đường nét từ bức ảnh, sử dụng [thuật toán Difference of Gaussians (DoG)](https://theailearner.com/2019/05/11/understanding-image-gradients/). Một cách đơn giản, ta lần lượt lấy đạo hàm theo 2 vùng khác nhau trên 1 bức ảnh về trung tâm (mỗi vùng là một ma trận vuông lẻ) , sau đó trừ 2 ma trận thu được để được đường nét của ảnh.
![DoG transfer](https://drive.google.com/file/d/1BZFXccflR2mAtlOIB1SFo407aF2Zr9jY/view?usp=sharing)

Ma trận cuối cùng thu được biểu diễn 6400 điểm ảnh là đặc trưng của bức ảnh.

b. Gaussian Naive Bayes

- Là một biến thể từ thuật toán Naive Bayes để giải quyết vấn đề các biến liên tục và sự rời rạc hóa các biến đó không giải quyết được vấn đề phân lớp.
- Dựa trên ý tưởng mọi biến trong tập đặc trưng đề có phân phối chuẩn, thay vì tính tỉ lệ xuất hiện của biến đó trong tập tri thức với điều kiện xác định, thuật toán này tìm cách chuẩn hóa các biên đó để coi nó là một phân phối chuẩn, sau đó với một biến đầu vào, thuật toán tìm xác suất xuất hiện của nó.




