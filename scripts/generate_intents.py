# -*- coding: utf-8 -*-
"""Generate high-quality intent keywords dictionary for all 222 leaf categories."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

INTENTS = {
    # ==================== ĂN UỐNG (FOOD & DRINK) ====================
    "23": "Quán cà phê, coffee shop, cf, cafe, đồ uống, trà, chỗ ngồi làm việc có wifi máy lạnh, tụ tập bạn bè, tán gẫu, học bài, cà phê muối, bạc xỉu, espresso, cold brew, cà phê đen đá, cà phê sữa, không gian yên tĩnh.",
    "24": "Quán trà sữa, tiệm trà, milk tea, boba, trà đào cam sả, trà mãng cầu, trà trái cây, trân châu đường đen, machiato, đồ uống ngọt, tụ tập học sinh sinh viên, mua mang đi.",
    "213": "Cửa hàng cháo, quán cháo, tiệm cháo, cháo sườn, cháo lòng, cháo vịt, cháo gà, cháo dinh dưỡng, cháo cá, đồ ăn ấm bụng, ăn nhẹ khi ốm bệnh, cháo thịt bằm, cháo ếch.",
    "218": "Quán ăn vặt, tiệm ăn vặt, xiên bẩn, bánh tráng trộn, cá viên chiên, nem chua rán, khoai tây chiên, trà chanh, chân gà sả tắc, tụ tập ăn vặt chiều, đồ nhâm nhi.",
    "53": "Nhà hàng BBQ, quán nướng, buffet nướng, thịt nướng than hoa, sườn nướng, bò nướng tảng, nướng ngói, nướng không khói, ăn nhậu liên hoan, tiệc nướng gia đình.",
    "54": "Nhà hàng Trung Hoa, quán món Hoa, dimsum, há cảo, xíu mại, mì vịt tiềm, vịt quay Bắc Kinh, cơm chiên Dương Châu, lẩu Tứ Xuyên cay nồng, ẩm thực Quảng Đông.",
    "55": "Nhà hàng chay, quán chay, cơm chay, buffet chay, món ăn thực vật, thuần chay, ăn chay ngày rằm mùng một, thanh đạm, rau củ quả, nấm tươi, đậu hũ.",
    "56": "Nhà hàng Ấn, quán món Ấn Độ, bánh naan, cà ri dê gà, cơm biryani, ẩm thực Halal, gia vị hồi quế, cà ri cay béo ngậy, tandoori.",
    "57": "Nhà hàng Nhật, quán món Nhật Bản, sushi, sashimi, cá hồi tươi, udon, tempura, cơm lươn, bento, ẩm thực Nhật Bản cao cấp và gia đình.",
    "58": "Cửa hàng bánh mì, tiệm bánh mì, bánh mì kẹp, bánh mì thịt, bánh mì chả lụa, bánh mì pate trứng ốp la, bánh mì que, bánh mì chảo, ăn sáng nhanh tiện lợi.",
    "59": "Nhà hàng Hàn Quốc, quán món Hàn, kimbap, tokbokki, mì cay Hàn Quốc, lẩu kim chi, cơm trộn bibimbap, gà rán sốt cay, ba chỉ nướng kim chi.",
    "60": "Nhà hàng Pizza, tiệm pizza, pizza hải sản, pizza xúc xích bò, mì ý spaghetti, pasta, phô mai kéo sợi, gà rán giòn, ăn nhanh gia đình bạn bè.",
    "61": "Nhà hàng Ramen, quán mì ramen, ramen tonkotsu, mì miso ramen, trứng ngâm tương, thịt chashu, mì udon, nước dùng hầm xương đậm đà kiểu Nhật.",
    "62": "Nhà hàng hải sản, quán hải sản tươi sống, cua ghẹ hấp, tôm hùm, ốc hương sốt bơ tỏi, sò điệp nướng mỡ hành, lẩu hải sản, hải sản bắt tại hồ.",
    "63": "Nhà hàng bít tết, quán steak, beefsteak, bò bít tết áp chảo, bò sốt tiêu đen, bò sốt nấm, khoai tây múi cau, rượu vang, hẹn hò lãng mạn.",
    "64": "Nhà hàng Thái, quán món Thái Lan, tomyum chua cay, pad thai mì xào, gỏi đu đủ som tum, lẩu Thái cay nồng, xôi xoài ngọt béo ngậy.",
    "65": "Nhà hàng Sushi, quán sushi, cơm cuộn nigiri maki, sashimi cá ngừ cá hồi bạch tuộc, wasabi gừng hồng, sushi tươi ngon chuẩn vị Nhật.",
    "66": "Nhà hàng món Việt, quán ăn Việt Nam, cơm niêu, phở bò gà, bún chả, nem rán chả giò, canh chua cá lóc, cá kho tộ, mâm cơm gia đình truyền thống.",
    "186": "Nhà hàng Pháp, ẩm thực Pháp, gan ngỗng foie gras, bò hầm vang đỏ, bánh crepe, rượu vang Pháp, không gian lãng mạn cao cấp, fine dining.",
    "179": "Nhà hàng Đài Loan, quán món Đài Loan, mì bò Đài Loan, bánh bao kẹp thịt guabao, đậu hũ thối, trà sữa trân châu chuẩn vị Đài.",
    "180": "Nhà hàng Brazil, quán nướng churrascaria, thịt xiên nướng kiểu Nam Mỹ, buffet thịt nướng tại bàn, thịt bò cừu heo nướng xiên.",
    "181": "Nhà hàng Ý, quán món Ý, pasta mì sốt cà chua thịt bằm bò băm bolognese, pizza nướng củi, bánh tiramisu, rượu vang Ý.",
    "20": "Nhà hàng, quán ăn, ẩm thực gia đình, đặt tiệc liên hoan, ăn trưa ăn tối, menu đa dạng món ngon, tiếp khách đối tác.",
    "21": "Tiệm bánh, bakery, tiệm bánh kem, tiệm bánh ngọt, bánh sinh nhật, bánh kem thôi nôi, bánh mì tươi, bánh croissant, pastry, bông lan trứng muối, nến sinh nhật.",
    "22": "Quán bar, quầy bar, cocktail bar, đồ uống có cồn, cocktail, mocktail, rượu mạnh, nghe nhạc chill, hẹn hò gặp gỡ ban đêm, không gian giải trí.",
    "25": "Cửa hàng kem, quán kem, tiệm kem lạnh, kem ốc quế, kem viên, kem tươi bingsu, lẩu kem, giải khát ngọt mát lạnh mùa hè.",
    "26": "Quán nhậu, pub, quán bia, bia hơi vỉa hè, lai rai, mồi nhậu, tụ tập bạn bè sau giờ làm, xem bóng đá, nhắm bia, đồ nướng nhắm, lẩu nhậu.",
    "75": "Thức ăn nhanh, fast food, gà rán giòn rụm, burger xúc xích hamburger khoai tây chiên, gọi món ăn liền mang đi, giao nhanh.",
    "168": "Quán ăn bình dân, quán cơm bình dân, cơm tấm sườn bì chả, quán cơm trưa văn phòng, ăn trưa no giá rẻ, bún nước bình dân ven đường.",
    "208": "Quán giải khát, tiệm nước ngọt, nước mía sầu riêng, nước dừa tươi, sinh tố bơ mãng cầu dâu tây, nước ép hoa quả cam dứa, trà tắc, đá me.",
    "209": "Nhà hàng Á Âu, ẩm thực fusion kết hợp, món âu món á, menu đa dạng hiện đại cho tiệc tùng gia đình bạn bè đối tác.",

    # ==================== SỨC KHỎE (HEALTH & WELLNESS) ====================
    "34": "Nhà thuốc, tiệm thuốc tây, pharmacy, hiệu thuốc, mua thuốc theo đơn, đau đầu, cảm sốt, sổ mũi, mua bông băng gạc y tế, thuốc sát trùng, thực phẩm chức năng, men tiêu hóa, thuốc giảm đau, nhiệt kế, que thử, vitamin.",
    "38": "Dịch vụ y tế sức khỏe, chăm sóc sức khỏe tại nhà, tư vấn sức khỏe, tiêm truyền tại nhà, điều dưỡng, dịch vụ y tế tư nhân.",
    "42": "Bệnh viện, hospital, cấp cứu 24/7, khám chữa bệnh, điều trị nội trú, viện đa khoa, khám chuyên khoa, chụp X-quang, xét nghiệm máu, tai nạn, ốm nặng.",
    "76": "Phòng khám nha khoa, tiệm răng, khám răng, nhổ răng khôn, sâu răng, đau răng buốt, lấy cao răng, niềng răng, bọc răng sứ, tẩy trắng răng, trám răng.",
    "77": "Bác sĩ, phòng khám bác sĩ tư, khám bệnh tư nhân, kê đơn thuốc, bác sĩ gia đình, tư vấn sức khỏe trực tiếp.",
    "78": "Spa, thẩm mỹ viện chăm sóc da, chăm sóc cơ thể, massage thư giãn, xông hơi, tẩy tế bào chết, trị mụn trẻ hóa làn da, làm đẹp thư thái.",
    "108": "Chăm sóc thú cưng, thú y, phòng khám thú y, tiêm phòng vaccine cho chó mèo, chữa bệnh thú cưng, spa tắm tỉa lông cho thú cưng, pet care.",
    "142": "Trạm y tế, trạm y tế phường xã, tiêm chủng mở rộng, khám sức khỏe ban đầu, sơ cấp cứu địa phương, cấp phát thuốc bảo hiểm.",
    "143": "Trung tâm y tế dự phòng, tiêm vắc xin vaccine, phòng chống dịch bệnh, kiểm tra vệ sinh an toàn dịch tễ, tiêm chủng phòng dại uốn ván.",
    "144": "Phòng khám đa khoa, phòng khám bệnh tổng quát, khám nội tiết tim mạch tiêu hóa, xét nghiệm siêu âm nhanh không phải chờ viện lớn.",
    "145": "Phòng khám chuyên khoa, khám tai mũi họng, khám da liễu, khám mắt kính, khám phụ khoa nam khoa, khám nhi khoa, bác sĩ chuyên khoa giỏi.",
    "146": "Thẩm mỹ viện, viện thẩm mỹ, phẫu thuật thẩm mỹ, nâng mũi cắt mí, tiêm filler botox, chăm sóc sắc đẹp công nghệ cao, làm đẹp toàn diện.",
    "147": "Y học cổ truyền, châm cứu, bấm huyệt, bốc thuốc bắc thuốc nam, đông y trị liệu, xoa bóp trị liệu đau lưng vai gáy, bắt mạch kê đơn.",
    "148": "Trung tâm dinh dưỡng, tư vấn chế độ ăn uống, khám suy dinh dưỡng béo phì ở trẻ em người lớn, thực đơn dinh dưỡng bệnh lý ăn kiêng.",
    "149": "Cơ sở vật lí trị liệu, phục hồi chức năng sau tai biến chấn thương gãy xương, kéo giãn cột sống, tập trị liệu thần kinh cơ xương khớp.",
    "150": "Cơ sở xét nghiệm, phòng xét nghiệm y khoa, lấy máu xét nghiệm tại nhà, xét nghiệm sinh hóa huyết học, xét nghiệm ADN tầm soát ung thư.",
    "170": "Mát xa, massage thư giãn, bấm huyệt trị liệu toàn thân vai gáy cổ, xông hơi đá nóng, massage chân foot massage, giải tỏa căng thẳng mệt mỏi.",
    "193": "Cơ sở dưỡng lão, viện dưỡng lão, nhà dưỡng lão, chăm sóc người cao tuổi, an dưỡng tuổi già có y tá bác sĩ túc trực chăm sóc 24/7.",
    "201": "Trung tâm y tế, trung tâm y tế quận huyện, khám chữa bệnh tuyến cơ sở, điều trị ngoại trú, tiêm chủng và bảo hiểm y tế cơ sở.",

    # ==================== MUA SẮM (SHOPPING & RETAIL) ====================
    "10": "Nhà sách, hiệu sách, tiệm sách, mua sách giáo khoa, truyện tranh, tiểu thuyết, sách tham khảo, văn phòng phẩm bút vở giấy kiểm tra, đồ chơi trẻ em.",
    "11": "Cửa hàng thời trang, shop quần áo, tiệm áo quần, thời trang nam nữ công sở trẻ trung, váy đầm áo thun sơ mi quần jean phụ kiện thời trang.",
    "12": "Cửa hàng máy tính, tiệm vi tính, laptop cá nhân, linh kiện PC máy bàn, chuột bàn phím tai nghe màn hình, sửa chữa bảo hành máy tính.",
    "13": "Cửa hàng tiện lợi, convenience store, cửa hàng 24h, mua đồ hộp mì gói nước uống bánh kẹo nhu yếu phẩm đồ ăn nhanh nóng hổi xuyên đêm.",
    "14": "Cửa hàng hoa, tiệm hoa tươi, shop hoa, hoa chúc mừng khai trương, hoa sinh nhật, bó hoa hồng tình yêu, hoa cưới cắm hoa theo yêu cầu.",
    "15": "Cửa hàng quà tặng, shop quà lưu niệm, gấu bông, sổ tay quà xinh, đồ decor trang trí, đồ kỷ niệm cho bạn bè người thân các dịp lễ.",
    "16": "Cửa hàng vật liệu xây dựng, đại lý xi măng cát đá gạch men, sắt thép xây dựng, sơn nước phụ gia chống thấm ống nước cho công trình.",
    "27": "Trung tâm thương mại, mua sắm giải trí phức hợp, shopping mall, tập trung các thương hiệu lớn thời trang ăn uống rạp chiếu phim vui chơi.",
    "28": "Siêu thị, mua sắm đồ gia dụng thực phẩm tươi sống bánh kẹo rau củ quả đồ dùng gia đình hàng ngày đầy đủ tiện lợi.",
    "29": "Chợ, chợ dân sinh, chợ truyền thống, mớ rau con cá mộc mạc tươi sống giá rẻ, mua bán thực phẩm mặc cả buổi sáng sớm.",
    "30": "Cửa hàng thịt, tiệm thịt tươi sống, thịt heo lợn bò gà tươi, sườn non thăn bò thịt sạch đạt chuẩn vệ sinh an toàn thực phẩm.",
    "31": "Cửa hàng sữa, đại lý sữa bỉm, sữa bột trẻ em, sữa tươi tiệt trùng, sữa chua dinh dưỡng cho mẹ bầu và bé sơ sinh.",
    "32": "Cửa hàng mỹ phẩm làm đẹp, shop mỹ phẩm xách tay chính hãng, son môi kem dưỡng da chống nắng serum phấn trang điểm nước hoa chính hãng.",
    "33": "Cửa hàng trang sức, tiệm vàng bạc, nhẫn kim cương vòng tay vàng cưới trang sức đá quý bông tai vàng 18k 24k bạc cao cấp.",
    "35": "Đại lý ô tô xe máy, showroom xe hơi xe máy, mua bán xe mới cũ chính hãng tư vấn trả góp bảo hiểm đăng ký xe biển số.",
    "36": "Cửa hàng giày dép, shop giày sneaker cao gót dép xăng đan giày thể thao nam nữ giày da công sở phụ kiện vệ sinh giày dép.",
    "37": "Cửa hàng điện thoại, shop smartphone điện thoại di động iPhone Samsung phụ kiện ốp lưng cáp sạc dán cường lực sửa chữa ép kính điện thoại.",
    "109": "Cửa hàng xe đạp, shop xe đạp thể thao xe đạp địa hình trẻ em mini phụ tùng mũ bảo hiểm đèn xe bảo dưỡng sửa chữa xe đạp.",
    "123": "Cửa hàng điện tử, thiết bị âm thanh loa kéo tai nghe bluetooth camera tivi thiết bị thông minh nhà ở phụ kiện điện tử số.",
    "124": "Cửa hàng dụng cụ, kim khí đồ nghề cơ khí ốc vít cờ lê máy khoan máy mài kìm tua vít đồ ngũ kim sửa chữa gia đình công trình.",
    "125": "Cửa hàng đồ gia dụng, xoong nồi chảo chống dính bát đĩa hộp nhựa chổi lau nhà kệ chén đồ dùng tiện ích nhà bếp gia đình phòng ngủ.",
    "126": "Cửa hàng trang sức, tiệm kim hoàn vàng bạc đá quý gia công trang sức nhẫn đôi nhẫn đính hôn nhẫn cưới cao cấp lấp lánh.",
    "127": "Cửa hàng rượu, tiệm rượu ngoại rượu vang vang đỏ vang trắng whisky cognac vodka đồ uống có cồn làm quà biếu hộp quà tết.",
    "128": "Cửa hàng thú cưng, pet shop thức ăn hạt cho chó mèo pate cát vệ sinh phụ kiện vòng cổ dây dắt chuồng đồ chơi thú cưng.",
    "129": "Cửa hàng giày dép, shop bán giày dép dép đi trong nhà dép tông giày búp bê giày chạy bộ êm chân giá tốt đa dạng mẫu mã.",
    "130": "Cửa hàng dụng cụ thể thao, đồ thể thao bóng đá bóng rổ vợt cầu lông tennis quần áo tập gym yoga thảm tập tạ tay găng tay thể thao.",
    "154": "Cửa hàng mỹ phẩm, cosmetics shop đồ skincare chăm sóc da mặt mặt nạ tẩy trang sữa rửa mặt dưỡng ẩm đồ trang điểm làm đẹp chính hãng.",
    "167": "Shophouse, nhà phố thương mại kết hợp ở và kinh doanh dịch vụ buôn bán tầng trệt tại các khu đô thị hiện đại sầm uất.",
    "169": "Trung tâm điện máy, siêu thị điện máy máy giặt tủ lạnh tivi máy lạnh điều hòa quạt máy đồ gia dụng lớn bảo hành chính hãng.",
    "174": "Thời trang trẻ em, shop quần áo trẻ em đồ sơ sinh váy bé gái quần áo bé trai chất vải mềm mát an toàn dễ thương cho bé.",
    "175": "Cửa hàng nội thất, bàn ghế sofa giường tủ quần áo kệ tivi bàn ăn gỗ decor trang trí nội thất nhà ở căn hộ văn phòng đẹp.",
    "176": "Cửa hàng tạp hóa, tiệm bách hóa nhỏ đầu ngõ, mua dầu ăn nước mắm bột ngọt gia vị giấy vệ sinh xà phòng kẹo bánh đồ dùng hàng ngày gần nhà.",
    "178": "Cửa hàng thực phẩm chức năng, thảo dược vitamin bổ sung khoáng chất collagen tăng đề kháng omega dầu cá hỗ trợ sức khỏe.",
    "182": "Cửa hàng bia, đại lý phân phối bia hơi bia lon thùng bia các loại giải khát nhậu liên hoan giá sỉ lẻ giao tận nơi.",
    "204": "Cửa hàng thủy hải sản tươi sống, cá tôm cua ghẹ mực tươi sống nhảy tanh tách sơ chế làm sạch tại chỗ bảo quản tươi ngon.",
    "205": "Cửa hàng rau sạch, thực phẩm hữu cơ organic, rau xanh củ quả tươi ngon không thuốc trừ sâu phục vụ bữa ăn an toàn gia đình.",
    "206": "Cửa hàng xổ số, đại lý vé số kiến thiết vietlott mega power cào trúng liền mua vận may đổi đời lĩnh thưởng uy tín.",
    "207": "Cửa hàng trái cây, sạp hoa quả nhập khẩu trái cây tươi ngon xoài bưởi sầu riêng cam táo nho giỏ quà trái cây chúc mừng.",
    "219": "Cửa hàng mắt kính, tiệm kính mắt đo mắt cắt kính cận thị viễn thị loạn thị kính râm thời trang gọng kính đẹp tròng chống ánh sáng xanh.",

    # ==================== DỊCH VỤ (SERVICES) ====================
    "52": "Cây ATM, máy rút tiền tự động, ngân hàng tự động, rút tiền thẻ, nạp tiền mặt, chuyển khoản, kiểm tra số dư thẻ ATM, banking 24/7.",
    "39": "Salon làm đẹp, tiệm thẩm mỹ làm đẹp da, nối mi uốn mi làm móng nail nghệ thuật chăm sóc sắc đẹp toàn diện cho phái nữ.",
    "40": "Tiệm tóc, salon tóc, quán cắt tóc nam nữ, tiệm barber, uốn nhuộm tóc tạo kiểu, phục hồi tóc gội đầu dưỡng sinh thư giãn.",
    "98": "Văn phòng tư vấn, công ty tư vấn tài chính du học định cư quản lý doanh nghiệp giải pháp đầu tư chuyên nghiệp tin cậy.",
    "99": "Chuyển phát nhanh, bưu cục chuyển phát hàng hóa thư tín gửi hàng bưu kiện ship COD giao nhận nhanh bưu phẩm trong ngoài nước.",
    "100": "Tiệm giặt ủi, giặt sấy tự động lấy ngay, tiệm giặt là đồ vest váy dạ hội áo len chăn ga mền gối thơm tho sạch sẽ.",
    "101": "Văn phòng luật sư, công ty luật tư vấn pháp lý tranh tụng hợp đồng hôn nhân gia đình đất đai hình sự công chứng pháp luật.",
    "102": "Thợ khóa, tiệm sửa khóa, cắt chìa khóa, đánh chìa mở khóa cửa bị kẹt, làm chìa khóa remote smartkey xe máy ô tô sửa ổ khóa tại nhà.",
    "103": "Dịch vụ chuyển nhà trọn gói, xe tải dọn nhà dọn văn phòng bốc xếp đồ đạc tháo lắp giường tủ an toàn nhanh chóng tiết kiệm.",
    "104": "Công ty bất động sản, môi giới nhà đất mua bán cho thuê căn hộ chung cư đất nền nhà phố mặt bằng kinh doanh ký gửi bất động sản.",
    "105": "Thợ may, tiệm may quần áo sửa quần áo lên gấu bóp eo may đo áo dài vest đầm dạ hội thời trang theo số đo cơ thể vừa vặn.",
    "106": "Dịch vụ viễn thông, cửa hàng viettel mobifone vinaphone hòa mạng đăng ký gói cước 4g 5g đổi sim esim lắp mạng internet wifi gia đình.",
    "107": "Đại lý du lịch, công ty tour lữ hành đặt tour du lịch trong nước quốc tế vé máy bay combo khách sạn visa du lịch nghỉ dưỡng.",
    "177": "Cửa hàng in ấn, tiệm photocopy in màu in tài liệu đồ án luận văn photo hồ sơ chứng minh căn cước đóng gáy bìa mạ vàng.",
    "183": "Studio chụp ảnh, tiệm chụp hình thẻ căn cước chụp ảnh chân dung nghệ thuật chụp kỷ yếu gia đình bé yêu ngoại cảnh đẹp.",
    "198": "Ảnh viện áo cưới, studio áo cưới trang điểm cô dâu chụp ảnh cưới trọn gói thuê váy cưới vest cưới phóng to ảnh cổng đẹp lung linh.",
    "5": "Tòa nhà văn phòng, cao ốc văn phòng cho thuê, tòa nhà làm việc trụ sở công ty tập đoàn không gian văn phòng làm việc chuyên nghiệp hiện đại.",

    # ==================== VẬN TẢI & GIAO THÔNG ====================
    "1": "Sân bay, cảng hàng không, phi trường, ga quốc nội quốc tế, đón trả khách đi máy bay, thủ tục bay vé máy bay hành lý gửi xe sân bay.",
    "17": "Điểm đậu đỗ GSM, bãi đỗ xe taxi xanh sm, điểm tập kết xe điện xanh di chuyển tiện lợi đón trả khách văn minh.",
    "18": "Trạm xe buýt, bến xe buýt đầu mối trung chuyển đón xe bus các tuyến trong thành phố chờ xe buýt công cộng giá rẻ.",
    "69": "Bưu điện, bưu cục chuyển phát bưu chính gửi thư gửi hàng chuyển tiền viễn thông đóng cước hành chính công cộng tiện lợi.",
    "111": "Ga tàu hỏa, nhà ga xe lửa ga đường sắt đón trả khách đi tàu hỏa mua vé tàu hỏa vận chuyển xe máy hàng hóa đường sắt.",
    "112": "Ga tàu điện, ga metro ga MRT đường sắt trên cao đón tàu đô thị di chuyển nhanh không tắc đường đi vé metro thẻ quẹt thẻ thông minh.",
    "113": "Bến phà, bến đò phà qua sông chở khách xe máy ô tô qua sông đường thủy ngắm cảnh sông nước tiện lợi giao thương đôi bờ.",
    "114": "Điểm dừng xe buýt, nhà chờ xe buýt ven đường đón xe bus ghé tuyến buýt dừng đón trả khách nhanh chóng thuận tiện.",
    "155": "Bến xe khách, bến xe liên tỉnh mua vé xe khách giường nằm xe limousine đi tỉnh đón người thân gửi nhận hàng hóa qua xe khách.",
    "156": "Bến cảng, cảng biển cảng sông đón tàu thuyền bốc dỡ container bốc dỡ hàng hóa du thuyền neo đậu tàu du lịch cập bến.",
    "171": "Trạm thu phí, trạm thu phí BOT tự động không dừng ETC nộp phí cầu đường kiểm soát phương tiện lưu thông đường cao tốc quốc lộ.",
    "211": "Điểm đón Xanh SM, vị trí đón xe taxi xe ôm điện xanh sm hẹn tài xế đón trả nhanh chóng thuận tiện đón xe công nghệ.",
    "238": "Nhà xe chành xe, điểm nhận gửi hàng chành xe vận chuyển hàng hóa đi các tỉnh gửi bưu kiện siêu tốc xe tải chở hàng tư nhân.",

    # ==================== Ô TÔ XE MÁY & NHIÊN LIỆU ====================
    "81": "Đại lý bán xe, salon ô tô xe máy mới cũ mua bán ký gửi xe hơi xe máy tư vấn tài chính mua xe trả góp thủ tục lăn bánh.",
    "82": "Cho thuê xe, dịch vụ thuê xe tự lái thuê xe du lịch 4 chỗ 7 chỗ 16 chỗ xe máy cho khách du lịch thuê xe hợp đồng đi tỉnh.",
    "83": "Tiệm sửa xe máy, gara sửa ô tô, vá xe vá lốp thay nhớt bơm lốp sửa xăm thay ắc quy sửa chữa phụ tùng hỏng hóc xe giữa đường cứu hộ xe.",
    "84": "Tiệm rửa xe, bãi rửa xe bọt tuyết dọn nội thất rửa khoang máy ô tô xe máy đánh bóng vệ sinh kính xe sạch đẹp nhanh chóng.",
    "85": "Cây xăng, trạm xăng dầu, đổ xăng xe máy bơm xăng đổ dầu diesel hết xăng trạm tiếp nhiên liệu ghé mua xăng dầu tiện lợi.",
    "86": "Bãi giữ xe, bãi đậu xe ô tô xe máy nhận trông giữ xe ngày đêm gửi xe qua đêm an toàn bãi xe có mái che bảo vệ 24/24.",
    "87": "Trạm dừng nghỉ ven đường cao tốc quốc lộ cho tài xế hành khách nghỉ chân đi vệ sinh ăn uống giải khát mua quà đặc sản.",

    # ==================== VINFAST & XE ĐIỆN ====================
    "220": "Tủ đổi pin, trạm đổi pin xe máy điện VinFast, thay pin xe điện lấy pin đầy nhanh chóng không cần chờ sạc tiện lợi tiết kiệm thời gian.",
    "222": "Showroom ô tô VinFast, đại lý xe ô tô điện VinFast VF3 VF5 VF6 VF7 VF8 VF9 trưng bày lái thử tư vấn mua xe điện VinFast chính hãng.",
    "223": "Showroom xe máy điện VinFast, đại lý xe máy điện Evo200 Feliz Klara Theon Vento trưng bày mua bán phụ kiện xe điện VinFast.",
    "224": "Xưởng dịch vụ ô tô VinFast, trung tâm bảo hành bảo dưỡng sửa chữa kiểm tra xe điện phụ tùng chính hãng ô tô VinFast chuyên nghiệp.",
    "225": "Xưởng dịch vụ ô tô đối tác VinFast, gara liên kết ủy quyền bảo dưỡng sửa chữa ô tô theo tiêu chuẩn chất lượng VinFast tin cậy.",
    "226": "Xưởng dịch vụ xe bus VinFast, trung tâm bảo trì bảo dưỡng xe buýt điện VinBus đảm bảo an toàn kỹ thuật vận hành công cộng sạch.",
    "227": "Xưởng dịch vụ xe máy điện VinFast, bảo trì thay thế phụ tùng động cơ acquy pin xe máy điện chính hãng kiểm tra lỗi xe điện.",
    "229": "Trạm sạc xe máy điện VinFast, trụ sạc pin cho xe máy điện cắm sạc tại chỗ tiện ích năng lượng xanh sạch cho xe điện.",
    "19": "Trạm sạc VinFast, trạm sạc xe ô tô điện VinFast, trụ sạc nhanh sạc siêu nhanh DC cắm sạc pin ô tô điện tại bãi đỗ xe trung tâm thương mại.",

    # ==================== VINGROUP ====================
    "231": "Vinhomes, khu đô thị Vinhomes khu dân cư cao cấp căn hộ biệt thự công viên hồ bơi môi trường sống tiện nghi an ninh khép kín.",
    "232": "Vinpearl, khu nghỉ dưỡng giải trí Vinpearl safari cáp treo công viên nước khách sạn resort vui chơi nghỉ mát gia đình đẳng cấp 5 sao.",
    "233": "Vincom, trung tâm thương mại Vincom Mega Mall Vincom Plaza mua sắm rạp chiếu phim khu vui chơi ẩm thực thời trang cao cấp sầm uất.",
    "234": "Vinschool, trường học liên cấp Vinschool mầm non tiểu học trung học giáo dục chuẩn quốc tế môi trường học tập hiện đại cho học sinh.",
    "235": "Vinmec, bệnh viện đa khoa quốc tế Vinmec khám chữa bệnh chất lượng cao trang thiết bị hiện đại dịch vụ y tế chuẩn quốc tế 5 sao.",
    "236": "VinUni, trường đại học VinUni khuôn viên đại học tinh hoa giảng dạy chuẩn quốc tế nghiên cứu công nghệ đào tạo nhân tài tương lai.",

    # ==================== NƠI LƯU TRÚ (ACCOMMODATION) ====================
    "3": "Khu dân cư, khu nhà ở, khu phố, khu dân sinh an cư lập nghiệp đường xá yên tĩnh nhà liền kề biệt thự cuộc sống thường nhật.",
    "7": "Khách sạn, hotel, phòng nghỉ qua đêm phòng đơn phòng đôi tiện nghi đầy đủ có ăn sáng tiếp tân 24/7 đặt phòng công tác du lịch.",
    "43": "Chung cư, tòa nhà chung cư, căn hộ cao tầng, khu căn hộ sinh sống thẻ từ thang máy an ninh bảo vệ gửi xe tầng hầm tiện ích hồ bơi.",
    "135": "Khu nghỉ dưỡng, resort ven biển trên núi không gian xanh trong lành hồ bơi view đẹp thư giãn cuối tuần kỳ nghỉ gia đình trọn gói.",
    "136": "Nhà trọ, phòng trọ cho thuê phòng sinh viên công nhân thuê trọ giá bình dân điện nước riêng phòng khép kín chỗ ở thuê trọ.",
    "137": "Nhà nghỉ, motel, phòng nghỉ theo giờ qua đêm giá rẻ kín đáo tiện nghi máy lạnh nước nóng giường nệm nghỉ ngơi nhanh.",
    "138": "Khách sạn nghỉ dưỡng, resort hotel cao cấp kết hợp spa hồ bơi ăn uống cảnh quan thiên nhiên tuyệt đẹp phục vụ kỳ nghỉ đẳng cấp.",
    "164": "Nhà ở, nhà riêng hộ gia đình khu phố cư trú riêng tư ấm cúng của người dân.",
    "166": "Khu biệt thự, villa sang trọng cảnh quan cây xanh an ninh bảo vệ nghiêm ngặt không gian sống thượng lưu yên tĩnh.",
    "172": "Cổng vào khu dân cư, chốt bảo vệ cổng chào an ninh khu phố barrier kiểm soát ra vào khu đô thị cư dân an toàn.",
    "189": "Ký túc xá, KTX sinh viên phòng ở tập thể sinh viên các trường đại học cao đẳng giường tầng chi phí thấp gần trường học bạn bè cùng phòng.",
    "212": "Nhà khách, nhà khách chính phủ cơ quan đoàn thể phòng nghỉ công vụ đón tiếp cán bộ công tác giá cả nhà nước tiện nghi cơ bản.",

    # ==================== CƠ SỞ GIÁO DỤC (EDUCATION) ====================
    "44": "Trường cấp 1, trường tiểu học lớp 1 đến lớp 5 cho học sinh nhỏ học chữ rèn luyện nề nếp bán trú học tập cơ sở giáo dục ban đầu.",
    "45": "Trường cấp 2, trường trung học cơ sở THCS lớp 6 đến lớp 9 học kiến thức phổ thông thiếu niên chuẩn bị thi vào trường cấp ba.",
    "46": "Trường cấp 3, trường trung học phổ thông THPT lớp 10 đến lớp 12 ôn thi tốt nghiệp đại học tuổi học trò năng động đèn sách.",
    "47": "Trường đại học, giảng đường đại học sinh viên học cử nhân kỹ sư thạc sĩ tiến sĩ khuôn viên ký túc xá nghiên cứu học thuật.",
    "48": "Trường cao đẳng, đào tạo thực hành nghề cao đẳng nghề thời gian ngắn thực tập doanh nghiệp cơ hội việc làm nhanh sau tốt nghiệp.",
    "49": "Trường mầm non, nhà trẻ trường mẫu giáo nuôi dạy chăm sóc trẻ em từ 1 đến 5 tuổi vui chơi học hát ăn bán trú giáo viên yêu trẻ.",
    "50": "Thư viện, không gian đọc sách mượn sách tra cứu tài liệu học tập nghiên cứu yên tĩnh bàn học máy lạnh wifi miễn phí.",
    "151": "Trường trung cấp, trường dạy nghề trung cấp nghề đào tạo kỹ thuật may điện nấu ăn sửa xe cơ khí cấp chứng chỉ nghề nghiệp.",
    "152": "Trung tâm ngoại ngữ, lớp học tiếng Anh tiếng Trung tiếng Hàn tiếng Nhật luyện thi IELTS TOEIC HSK giao tiếp ngoại ngữ hiệu quả.",
    "153": "Trường dạy lái xe, trung tâm đào tạo sát hạch lái xe ô tô hạng B1 B2 bằng C thi bằng lái xe máy A1 A2 sân tập lái uy tín.",
    "165": "Trường liên cấp, hệ thống trường quốc tế tư thục từ mầm non tiểu học đến trung học cơ sở phổ thông chương trình song ngữ hiện đại.",
    "195": "Trường dự bị đại học, khóa học chuẩn bị kiến thức tiếng Anh bổ trợ kỹ năng cho học sinh sinh viên trước khi vào đại học chính quy du học.",
    "203": "Trung tâm huấn luyện, cơ sở đào tạo bồi dưỡng kỹ năng nghiệp vụ chuyên môn quân sự thể thao cho cán bộ học viên nâng cao năng lực.",

    # ==================== NGOÀI TRỜI & GIẢI TRÍ ====================
    "74": "Công viên, vườn hoa không gian xanh công cộng đi dạo tập thể dục chạy bộ hóng mát bóng râm ghế đá cho người lớn trẻ em vui chơi.",
    "88": "Trung tâm giải trí, khu vui chơi trong nhà máy gắp thú đua xe bắn súng game xèng bowling giải trí cuối tuần cùng gia đình bạn bè.",
    "89": "Công viên giải trí, theme park trò chơi cảm giác mạnh tàu lượn siêu tốc đu quay vòng xoay khổng lồ tham quan vui nhộn sôi động.",
    "90": "Địa điểm tổ chức sự kiện, trung tâm hội nghị tiệc gala tất niên hội thảo biểu diễn sân khấu âm thanh ánh sáng chuyên nghiệp.",
    "91": "Di tích lịch sử, danh lam thắng cảnh đền đài thành cổ công trình văn hóa lâu đời tham quan tìm hiểu lịch sử cội nguồn dân tộc.",
    "92": "Rạp chiếu phim, rạp xem phim rạp cinema bắp nước bỏng ngô phim bom tấn chiếu rạp màn hình lớn âm thanh vòm hẹn hò cuối tuần.",
    "93": "Công viên quốc gia, vườn quốc gia khu bảo tồn thiên nhiên rừng rậm sinh thái trekking ngắm cảnh hoang dã cắm trại thám hiểm.",
    "94": "Hộp đêm, night club quầy bar nhạc sống DJ quẩy đêm nhảy múa cocktail ánh sáng sôi động giải trí về đêm của giới trẻ.",
    "95": "Khu du lịch, danh thắng tham quan trải nghiệm checkin chụp ảnh nghỉ dưỡng tham quan phong cảnh đẹp cuối tuần cho du khách.",
    "96": "Trung tâm tiệc cưới, sảnh cưới nhà hàng tiệc cưới hôn lễ cô dâu chú rể đặt bàn tiệc rượu mừng ngày cưới hoành tráng ấm cúng.",
    "97": "Sở thú, thảo cầm viên vườn bách thú tham quan động vật hoang dã hươu cao cổ voi hổ khỉ chim chóc cho trẻ nhỏ học tập khám phá.",
    "163": "Nhà văn hoá, nhà sinh hoạt cộng đồng điểm họp tổ dân phố giao lưu văn nghệ thể thao câu lạc bộ người cao tuổi khu dân cư.",
    "187": "Câu lạc bộ, hội nhóm sinh hoạt chung sở thích khiêu vũ võ thuật cờ vua hội họa giao lưu kết nối bạn bè cùng đam mê.",
    "188": "Karaoke, quán hát karaoke phòng hát cách âm âm thanh chất lượng cao micro màn hình chọn bài tụ tập bạn bè ca hát tiệc tùng xả stress.",
    "190": "Quảng trường, không gian công cộng rộng lớn trung tâm thành phố đi dạo thả diều ngắm đài phun nước tụ họp lễ hội pháo hoa đêm.",
    "199": "Công viên nước, khu trượt nước hồ tạo sóng bể bơi trò chơi máng trượt nước giải nhiệt mùa hè vui chơi thỏa thích cùng bạn bè gia đình.",
    "237": "Quán bida, clb bida billiards hall bàn bida libre carom lỗ snooker tụ tập bạn bè so cơ giải trí thư giãn điều hòa mát mẻ.",
    "239": "Khu cắm trại, bãi camping glamping dựng lều đốt lửa trại tiệc nướng BBQ ngoài trời ngắm sao hòa mình vào thiên nhiên cuối tuần.",
    "240": "Nhà vệ sinh công cộng, toilet WC công cộng sạch sẽ phục vụ người đi đường du khách giải quyết nhu cầu vệ sinh thiết yếu cá nhân.",
    "241": "Điểm dừng chân, trạm nghỉ ngơi giải khát vệ sinh mua quà bánh nghỉ ngơi lấy lại sức trong chuyến hành trình đường dài.",

    # ==================== THỂ THAO (SPORTS) ====================
    "116": "Trung tâm thể dục thể thao, khu thể thao phức hợp tập gym bóng chuyền cầu lông bóng bàn rèn luyện nâng cao sức khỏe cộng đồng.",
    "117": "Sân golf, câu lạc bộ golf sân cỏ 18 lỗ gậy golf xe điện caddie tập đánh golf đẳng cấp thể thao thượng lưu giao lưu đối tác.",
    "118": "Phòng tập gym, phòng thể hình máy tập tạ máy chạy bộ huấn luyện viên PT cá nhân tăng cơ giảm mỡ rèn luyện vóc dáng săn chắc.",
    "119": "Sân chơi trẻ em, xích đu cầu trượt bập bênh khu vận động ngoài trời an toàn cho bé vui chơi chạy nhảy kết bạn.",
    "120": "Khu phức hợp thể thao, cụm sân bóng đá sân tennis hồ bơi nhà thi đấu đa năng tập luyện và thi đấu thể thao chuyên nghiệp.",
    "121": "Sân vận động, khán đài lớn sân cỏ bóng đá thi đấu thể thao tổ chức đại nhạc hội giải đấu thể thao đông đảo người hâm mộ cổ vũ.",
    "122": "Hồ bơi, bể bơi công cộng bể bơi bốn mùa làn bơi học bơi rèn luyện bơi lội giải nhiệt mùa hè huấn luyện viên dạy bơi an toàn.",
    "202": "Nhà thi đấu, nhà thi đấu đa năng trong nhà sàn gỗ khán đài thi đấu bóng rổ cầu lông võ thuật bóng bàn quy mô lớn mát mẻ.",

    # ==================== ĐỊA ĐIỂM THỜ CÚNG & TÂM LINH ====================
    "140": "Nhà thờ, giáo xứ nhà thờ công giáo đi lễ chúa nhật cầu nguyện thánh lễ đức mẹ chúa giêsu tiếng chuông nhà thờ thanh bình.",
    "141": "Chùa, chùa chiền đi chùa dâng hương lạy phật cầu an thanh tịnh vãn cảnh chùa ngày rằm mùng một ăn cơm chay cửa phật từ bi.",
    "157": "Đình làng, đình miếu thờ thành hoàng làng di tích kiến trúc cổ truyền hội đình làng gìn giữ phong tục tập quán truyền thống.",
    "158": "Đền, đền thờ danh nhân anh hùng dân tộc dâng hương tưởng nhớ công đức tiền nhân chiêm bái cầu phúc lộc bình an.",
    "159": "Miếu, miếu thờ miếu bà miếu thần linh thổ địa thắp nhang cầu may khấn vái tâm linh phong tục tín ngưỡng dân gian trang nghiêm.",
    "160": "Thánh thất, thánh thất Cao Đài đạo Cao Đài chiêm bái biểu tượng thiên nhãn nghi lễ tôn giáo trang nghiêm áo trắng thanh sạch.",
    "161": "Nhà thờ Hồi giáo, thánh đường Hồi giáo mosque tín đồ đạo Hồi cầu nguyện thứ sáu hướng về Mecca tôn kính Allah tôn giáo nghiêm cẩn.",
    "162": "Nhà thờ họ, từ đường nhà thờ tổ gia tộc con cháu tề tựu cúng giỗ tổ tiên ngày lễ tết ghi nhớ công ơn cội nguồn gia đình dòng họ.",
    "184": "Nghĩa trang, hoa viên nghĩa trang nơi an nghỉ viếng mộ người đã khuất thắp nhang tảo mộ thanh minh xây cất phần mộ trang nghiêm.",
    "196": "Nhà tang lễ, cơ sở tổ chức tang lễ lễ viếng lễ truy điệu đám tang tiễn đưa người đã khuất chu đáo thành kính chia buồn cùng gia quyến.",
    "197": "Cơ sở hỏa táng, đài hóa thân hoàn vũ nơi thực hiện hỏa thiêu lưu giữ tro cốt trang nghiêm văn minh sạch sẽ chia buồn gia đình.",

    # ==================== CHÍNH QUYỀN & HÀNH CHÍNH ====================
    "68": "Trụ sở công an, đồn công an cơ quan công an phường xã trình báo mất đồ an ninh trật tự làm căn cước công dân hộ khẩu tạm trú tạm vắng.",
    "70": "Văn phòng chính quyền địa phương, ủy ban nhân dân UBND phường xã thị trấn giải quyết thủ tục hành chính giấy tờ công chứng chứng thực hộ tịch.",
    "71": "Trạm cứu hỏa, đội phòng cháy chữa cháy PCCC ứng cứu khẩn cấp dập lửa chữa cháy cứu nạn cứu hộ xe chữa cháy chuyên dụng 24/7.",
    "72": "Đại sứ quán, lãnh sự quán cơ quan ngoại giao xin visa thị thực cấp hộ chiếu hỗ trợ công dân nước ngoài quan hệ ngoại giao quốc tế.",
    "73": "Tòa án, tòa án nhân dân xét xử tranh chấp kiện tụng tố tụng pháp luật phân xử dân sự hình sự bảo vệ công lý quyền lợi hợp pháp.",
    "173": "Cơ quan nhà nước, cơ quan hành chính công sở phục vụ nhân dân giải quyết hồ sơ giấy tờ thuế bảo hiểm quy hoạch quản lý nhà nước.",

    # ==================== VĂN HÓA & NGHỆ THUẬT ====================
    "132": "Phòng trưng bày nghệ thuật, art gallery triển lãm tranh tượng tác phẩm nghệ thuật thưởng lãm hội họa điêu khắc giao lưu nghệ sĩ.",
    "133": "Bảo tàng, museum bảo tồn hiện vật lịch sử cổ vật văn hóa triển lãm học tập nghiên cứu mở mang kiến thức di sản văn hóa quý giá.",
    "134": "Nhà hát, sân khấu kịch biểu diễn nghệ thuật ca múa nhạc hòa nhạc giao hưởng kịch nói tuồng chèo thưởng thức nghệ thuật đỉnh cao.",
    "194": "Công trình kỷ niệm, tượng đài bia tưởng niệm lịch sử ghi dấu chiến công sự kiện trọng đại điểm tham quan tự hào dân tộc.",
    "215": "Trung tâm hội nghị triển lãm, convention center triển lãm thương mại hội chợ quốc tế quy mô lớn sự kiện doanh nghiệp hội thảo cấp cao.",

    # ==================== CÔNG TY, HẠ TẦNG & KHÁC ====================
    "191": "Cơ sở từ thiện, mái ấm nhà tình thương trung tâm bảo trợ xã hội nhận quyên góp hỗ trợ người già neo đơn trẻ em mồ côi khó khăn thiện nguyện.",
    "192": "Văn phòng công ty, trụ sở doanh nghiệp văn phòng làm việc kinh doanh giao dịch đối tác nơi làm việc nhân viên văn phòng.",
    "200": "Cơ sở sản xuất, nhà xưởng nhà máy chế biến gia công sản xuất hàng hóa công nghiệp kho bãi máy móc hoạt động sản xuất quy mô.",
    "214": "Cơ sở hạ tầng kỹ thuật, trạm biến áp trạm bơm nhà máy nước xử lý rác thải công trình kỹ thuật phục vụ vận hành đô thị liên tục.",
    "216": "A80, địa điểm chuyên dụng cơ sở kỹ thuật theo phân loại hệ thống phục vụ mục đích kiểm tra và vận hành chuyên biệt.",
    "217": "Công ty du lịch, văn phòng lữ hành tư vấn bán tour du lịch trong và ngoài nước đặt vé máy bay khách sạn hỗ trợ làm visa du lịch trọn gói.",
}


def main():
    tax_path = Path("data/taxonomies/place_types.poc.v1.json")
    tax = json.loads(tax_path.read_text(encoding="utf-8"))
    leaves = [x for x in tax["items"] if not any(y.get("parent_id") == x["id"] for y in tax["items"])]
    leaf_ids = {str(x["id"]) for x in leaves}
    
    missing = leaf_ids - set(INTENTS.keys())
    extra = set(INTENTS.keys()) - leaf_ids
    
    if missing:
        print(f"ERROR: Missing {len(missing)} categories: {sorted(missing, key=int)}")
        sys.exit(1)
    if extra:
        print(f"ERROR: Extra {len(extra)} categories not in taxonomy: {sorted(extra, key=int)}")
        sys.exit(1)
        
    print(f"SUCCESS: Exact match for all {len(INTENTS)} leaf categories!")
    
    output_path = Path("data/taxonomies/place_type_intent_keywords.v1.json")
    output_payload = {
        "version": "1.0",
        "model_target": "BAAI/bge-m3",
        "description": "Rich intent keywords mapping for semantic retrieval",
        "total_categories": len(INTENTS),
        "items": {k: {"search_text_vi": v} for k, v in sorted(INTENTS.items(), key=lambda x: int(x[0]))}
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(INTENTS)} rich intents to {output_path.resolve()}")


if __name__ == "__main__":
    main()
