select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5747'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17243'
  and agency_id_id= '17243'
  and notice_id= '17243'
  and route_id= '17243';
select COUNT(*)
from dv.notes_message
where user_id='13424'
  and agency_id_id= '13424'
  and notice_id= '13424'
  and route_id= '13424';
select COUNT(*)
from dv.notes_message
where user_id='6289'
  and agency_id_id= '6289'
  and notice_id= '6289'
  and route_id= '6289';
select user_id
from m.agency
where valid_now=12806
  and agency_id_id= '6274';
select COUNT(*)
from dv.notes_message
where user_id='14804'
  and agency_id_id= '14804'
  and notice_id= '14804'
  and route_id= '14804';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14639'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15972'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10062'
  and valid_now=6470;
select agency_id
from m.agency
where agency_id_id= '10927'
  and valid_now=19912;
select user_id
from m.agency
where valid_now=2829
  and agency_id_id= '8581';
select COUNT(*)
from dv.notes_message
where user_id='19439'
  and agency_id_id= '19439'
  and notice_id= '19439'
  and route_id= '19439';
select agency_id
from m.agency
where agency_id_id= '14677'
  and valid_now=3890;
select COUNT(*)
from dv.notes_message
where user_id='13251'
  and agency_id_id= '13251'
  and notice_id= '13251'
  and route_id= '13251';
select user_id
from m.agency
where valid_now=14459
  and agency_id_id= '13250';
select user_id
from m.agency
where valid_now=19174
  and agency_id_id= '8023';
select user_id
from m.agency
where valid_now=19019
  and agency_id_id= '18247';
select COUNT(*)
from dv.notes_message
where user_id='4325'
  and agency_id_id= '4325'
  and notice_id= '4325'
  and route_id= '4325';
select user_id
from m.agency
where valid_now=17777
  and agency_id_id= '3568';
select user_id
from m.agency
where valid_now=19198
  and agency_id_id= '9198';
select user_id
from m.agency
where valid_now=1882
  and agency_id_id= '13385';
select agency_id
from m.agency
where agency_id_id= '5350'
  and valid_now=1063;
select user_id
from m.agency
where valid_now=17496
  and agency_id_id= '14204';
select a.agency_timezone
from m.agency a
where a.agency_id = '2436';
select agency_id
from m.agency
where agency_id_id= '6076'
  and valid_now=5874;
select agency_id
from m.agency
where agency_id_id= '15051'
  and valid_now=6084;
select COUNT(*)
from dv.notes_message
where user_id='2415'
  and agency_id_id= '2415'
  and notice_id= '2415'
  and route_id= '2415';
select COUNT(*)
from dv.notes_message
where user_id='17703'
  and agency_id_id= '17703'
  and notice_id= '17703'
  and route_id= '17703';
select agency_id
from m.agency
where agency_id_id= '8614'
  and valid_now=1491;
select COUNT(*)
from dv.notes_message
where user_id='6359'
  and agency_id_id= '6359'
  and notice_id= '6359'
  and route_id= '6359';
select a.agency_timezone
from m.agency a
where a.agency_id = '1741';
select agency_id
from m.agency
where agency_id_id= '8684'
  and valid_now=5280;
select agency_id
from m.agency
where agency_id_id= '13625'
  and valid_now=12050;
select user_id
from m.agency
where valid_now=8664
  and agency_id_id= '7277';
select a.agency_timezone
from m.agency a
where a.agency_id = '15533';
select user_id
from m.agency
where valid_now=12627
  and agency_id_id= '19464';
select agency_id
from m.agency
where agency_id_id= '17410'
  and valid_now=19935;
select COUNT(*)
from dv.notes_message
where user_id='13165'
  and agency_id_id= '13165'
  and notice_id= '13165'
  and route_id= '13165';
select a.agency_timezone
from m.agency a
where a.agency_id = '12390';
select agency_id
from m.agency
where agency_id_id= '1417'
  and valid_now=15554;
select user_id
from m.agency
where valid_now=10120
  and agency_id_id= '1543';
select COUNT(*)
from dv.notes_message
where user_id='3220'
  and agency_id_id= '3220'
  and notice_id= '3220'
  and route_id= '3220';
select agency_id
from m.agency
where agency_id_id= '19743'
  and valid_now=5702;
select user_id
from m.agency
where valid_now=7317
  and agency_id_id= '4526';
select COUNT(*)
from dv.notes_message
where user_id='4280'
  and agency_id_id= '4280'
  and notice_id= '4280'
  and route_id= '4280';
select agency_id
from m.agency
where agency_id_id= '18114'
  and valid_now=4607;
select agency_id
from m.agency
where agency_id_id= '9626'
  and valid_now=2309;
select user_id
from m.agency
where valid_now=6222
  and agency_id_id= '11004';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1862'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14868'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16087'
  and valid_now=9474;
select user_id
from m.agency
where valid_now=11871
  and agency_id_id= '14281';
select COUNT(*)
from dv.notes_message
where user_id='1293'
  and agency_id_id= '1293'
  and notice_id= '1293'
  and route_id= '1293';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1767'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '990'
  and valid_now=671;
select user_id
from m.agency
where valid_now=3244
  and agency_id_id= '12224';
select agency_id
from m.agency
where agency_id_id= '8440'
  and valid_now=19897;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4682'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12379'
  and agency_id_id= '12379'
  and notice_id= '12379'
  and route_id= '12379';
select agency_id
from m.agency
where agency_id_id= '13380'
  and valid_now=9707;
select user_id
from m.agency
where valid_now=15923
  and agency_id_id= '1201';
select COUNT(*)
from dv.notes_message
where user_id='1475'
  and agency_id_id= '1475'
  and notice_id= '1475'
  and route_id= '1475';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11292'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2788'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11219'
  and agency_id_id= '11219'
  and notice_id= '11219'
  and route_id= '11219';
select COUNT(*)
from dv.notes_message
where user_id='15779'
  and agency_id_id= '15779'
  and notice_id= '15779'
  and route_id= '15779';
select agency_id
from m.agency
where agency_id_id= '7306'
  and valid_now=17301;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5306'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8303'
  and valid_now=12525;
select user_id
from m.agency
where valid_now=12844
  and agency_id_id= '11640';
select user_id
from m.agency
where valid_now=17418
  and agency_id_id= '11374';
select COUNT(*)
from dv.notes_message
where user_id='1317'
  and agency_id_id= '1317'
  and notice_id= '1317'
  and route_id= '1317';
select agency_id
from m.agency
where agency_id_id= '9050'
  and valid_now=3897;
select agency_id
from m.agency
where agency_id_id= '1811'
  and valid_now=9578;
select COUNT(*)
from dv.notes_message
where user_id='19813'
  and agency_id_id= '19813'
  and notice_id= '19813'
  and route_id= '19813';
select user_id
from m.agency
where valid_now=8182
  and agency_id_id= '18440';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16570'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17249'
  and valid_now=18811;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4421'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1785
  and agency_id_id= '7267';
select user_id
from m.agency
where valid_now=16296
  and agency_id_id= '2217';
select COUNT(*)
from dv.notes_message
where user_id='679'
  and agency_id_id= '679'
  and notice_id= '679'
  and route_id= '679';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4764'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8243'
  and valid_now=2659;
select user_id
from m.agency
where valid_now=16434
  and agency_id_id= '8395';
select COUNT(*)
from dv.notes_message
where user_id='4062'
  and agency_id_id= '4062'
  and notice_id= '4062'
  and route_id= '4062';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5347'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13695'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=638
  and agency_id_id= '12275';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7332'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19446'
  and valid_now=10001;
select user_id
from m.agency
where valid_now=10585
  and agency_id_id= '1282';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4432'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5719'
  and valid_now=8994;
select agency_id
from m.agency
where agency_id_id= '4782'
  and valid_now=18271;
select user_id
from m.agency
where valid_now=12831
  and agency_id_id= '17487';
select COUNT(*)
from dv.notes_message
where user_id='16690'
  and agency_id_id= '16690'
  and notice_id= '16690'
  and route_id= '16690';
select COUNT(*)
from dv.notes_message
where user_id='18648'
  and agency_id_id= '18648'
  and notice_id= '18648'
  and route_id= '18648';
select user_id
from m.agency
where valid_now=12943
  and agency_id_id= '4086';
select agency_id
from m.agency
where agency_id_id= '16408'
  and valid_now=1064;
select user_id
from m.agency
where valid_now=661
  and agency_id_id= '8140';
select agency_id
from m.agency
where agency_id_id= '5514'
  and valid_now=6813;
select COUNT(*)
from dv.notes_message
where user_id='19026'
  and agency_id_id= '19026'
  and notice_id= '19026'
  and route_id= '19026';
select COUNT(*)
from dv.notes_message
where user_id='13403'
  and agency_id_id= '13403'
  and notice_id= '13403'
  and route_id= '13403';
select COUNT(*)
from dv.notes_message
where user_id='17682'
  and agency_id_id= '17682'
  and notice_id= '17682'
  and route_id= '17682';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6537'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14918'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2952'
  and valid_now=19758;
select agency_id
from m.agency
where agency_id_id= '6561'
  and valid_now=19399;
select user_id
from m.agency
where valid_now=5477
  and agency_id_id= '6929';
select user_id
from m.agency
where valid_now=2630
  and agency_id_id= '11704';
select user_id
from m.agency
where valid_now=4129
  and agency_id_id= '9483';
select COUNT(*)
from dv.notes_message
where user_id='11299'
  and agency_id_id= '11299'
  and notice_id= '11299'
  and route_id= '11299';
select COUNT(*)
from dv.notes_message
where user_id='1079'
  and agency_id_id= '1079'
  and notice_id= '1079'
  and route_id= '1079';
select COUNT(*)
from dv.notes_message
where user_id='4853'
  and agency_id_id= '4853'
  and notice_id= '4853'
  and route_id= '4853';
select COUNT(*)
from dv.notes_message
where user_id='2714'
  and agency_id_id= '2714'
  and notice_id= '2714'
  and route_id= '2714';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6214'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2764'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16915'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4401
  and agency_id_id= '6564';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16624'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '820'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3362'
  and valid_now=15924;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13440'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5249'
  and agency_id_id= '5249'
  and notice_id= '5249'
  and route_id= '5249';
select COUNT(*)
from dv.notes_message
where user_id='8172'
  and agency_id_id= '8172'
  and notice_id= '8172'
  and route_id= '8172';
select user_id
from m.agency
where valid_now=227
  and agency_id_id= '1730';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12419'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10634'
  and valid_now=12965;
select user_id
from m.agency
where valid_now=14872
  and agency_id_id= '2709';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '799'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18391'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16580'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10864'
  and valid_now=2636;
select user_id
from m.agency
where valid_now=2567
  and agency_id_id= '1602';
select COUNT(*)
from dv.notes_message
where user_id='8636'
  and agency_id_id= '8636'
  and notice_id= '8636'
  and route_id= '8636';
select COUNT(*)
from dv.notes_message
where user_id='19191'
  and agency_id_id= '19191'
  and notice_id= '19191'
  and route_id= '19191';
select agency_id
from m.agency
where agency_id_id= '9854'
  and valid_now=6979;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5169'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14709'
  and agency_id_id= '14709'
  and notice_id= '14709'
  and route_id= '14709';
select COUNT(*)
from dv.notes_message
where user_id='433'
  and agency_id_id= '433'
  and notice_id= '433'
  and route_id= '433';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17343'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15291'
  and valid_now=529;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17589'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5594'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6841'
  and valid_now=18692;
select agency_id
from m.agency
where agency_id_id= '18469'
  and valid_now=13768;
select user_id
from m.agency
where valid_now=15052
  and agency_id_id= '19089';
select agency_id
from m.agency
where agency_id_id= '14959'
  and valid_now=14410;
select COUNT(*)
from dv.notes_message
where user_id='5914'
  and agency_id_id= '5914'
  and notice_id= '5914'
  and route_id= '5914';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10936'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14130'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11442'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16234'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15548'
  and agency_id_id= '15548'
  and notice_id= '15548'
  and route_id= '15548';
select agency_id
from m.agency
where agency_id_id= '19176'
  and valid_now=10645;
select user_id
from m.agency
where valid_now=10749
  and agency_id_id= '344';
select user_id
from m.agency
where valid_now=9413
  and agency_id_id= '14079';
select user_id
from m.agency
where valid_now=1402
  and agency_id_id= '16721';
select COUNT(*)
from dv.notes_message
where user_id='15461'
  and agency_id_id= '15461'
  and notice_id= '15461'
  and route_id= '15461';
select agency_id
from m.agency
where agency_id_id= '15352'
  and valid_now=8188;
select COUNT(*)
from dv.notes_message
where user_id='4297'
  and agency_id_id= '4297'
  and notice_id= '4297'
  and route_id= '4297';
select user_id
from m.agency
where valid_now=717
  and agency_id_id= '19852';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9295'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18576'
  and valid_now=3355;
select COUNT(*)
from dv.notes_message
where user_id='10396'
  and agency_id_id= '10396'
  and notice_id= '10396'
  and route_id= '10396';
select COUNT(*)
from dv.notes_message
where user_id='14509'
  and agency_id_id= '14509'
  and notice_id= '14509'
  and route_id= '14509';
select agency_id
from m.agency
where agency_id_id= '8005'
  and valid_now=16910;
select user_id
from m.agency
where valid_now=3997
  and agency_id_id= '10556';
select agency_id
from m.agency
where agency_id_id= '3215'
  and valid_now=12736;
select user_id
from m.agency
where valid_now=11132
  and agency_id_id= '5496';
select agency_id
from m.agency
where agency_id_id= '16958'
  and valid_now=1359;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18094'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17743'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9386'
  and valid_now=7294;
select COUNT(*)
from dv.notes_message
where user_id='3575'
  and agency_id_id= '3575'
  and notice_id= '3575'
  and route_id= '3575';
select COUNT(*)
from dv.notes_message
where user_id='5416'
  and agency_id_id= '5416'
  and notice_id= '5416'
  and route_id= '5416';
select COUNT(*)
from dv.notes_message
where user_id='19397'
  and agency_id_id= '19397'
  and notice_id= '19397'
  and route_id= '19397';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12283'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11894'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13264'
  and valid_now=18079;
select a.agency_timezone
from m.agency a
where a.agency_id = '10950';
select a.agency_timezone
from m.agency a
where a.agency_id = '11933';
select agency_id
from m.agency
where agency_id_id= '14280'
  and valid_now=2084;
select COUNT(*)
from dv.notes_message
where user_id='10277'
  and agency_id_id= '10277'
  and notice_id= '10277'
  and route_id= '10277';
select COUNT(*)
from dv.notes_message
where user_id='3086'
  and agency_id_id= '3086'
  and notice_id= '3086'
  and route_id= '3086';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4803'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2677
  and agency_id_id= '5302';
select COUNT(*)
from dv.notes_message
where user_id='11456'
  and agency_id_id= '11456'
  and notice_id= '11456'
  and route_id= '11456';
select COUNT(*)
from dv.notes_message
where user_id='18398'
  and agency_id_id= '18398'
  and notice_id= '18398'
  and route_id= '18398';
select user_id
from m.agency
where valid_now=16972
  and agency_id_id= '6832';
select agency_id
from m.agency
where agency_id_id= '16705'
  and valid_now=9079;
select user_id
from m.agency
where valid_now=851
  and agency_id_id= '16416';
select a.agency_timezone
from m.agency a
where a.agency_id = '15443';
select agency_id
from m.agency
where agency_id_id= '18460'
  and valid_now=4436;
select a.agency_timezone
from m.agency a
where a.agency_id = '17827';
select agency_id
from m.agency
where agency_id_id= '326'
  and valid_now=3600;
select agency_id
from m.agency
where agency_id_id= '14093'
  and valid_now=17001;
select COUNT(*)
from dv.notes_message
where user_id='18290'
  and agency_id_id= '18290'
  and notice_id= '18290'
  and route_id= '18290';
select a.agency_timezone
from m.agency a
where a.agency_id = '5406';
select user_id
from m.agency
where valid_now=9537
  and agency_id_id= '6750';
select COUNT(*)
from dv.notes_message
where user_id='14815'
  and agency_id_id= '14815'
  and notice_id= '14815'
  and route_id= '14815';
select COUNT(*)
from dv.notes_message
where user_id='14403'
  and agency_id_id= '14403'
  and notice_id= '14403'
  and route_id= '14403';
select COUNT(*)
from dv.notes_message
where user_id='16270'
  and agency_id_id= '16270'
  and notice_id= '16270'
  and route_id= '16270';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3848'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15922'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14032'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19940'
  and valid_now=1429;
select agency_id
from m.agency
where agency_id_id= '13347'
  and valid_now=3043;
select agency_id
from m.agency
where agency_id_id= '16951'
  and valid_now=11840;
select agency_id
from m.agency
where agency_id_id= '5296'
  and valid_now=14384;
select agency_id
from m.agency
where agency_id_id= '16277'
  and valid_now=1119;
select COUNT(*)
from dv.notes_message
where user_id='14540'
  and agency_id_id= '14540'
  and notice_id= '14540'
  and route_id= '14540';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '30'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10241
  and agency_id_id= '12540';
select user_id
from m.agency
where valid_now=16983
  and agency_id_id= '14377';
select COUNT(*)
from dv.notes_message
where user_id='4526'
  and agency_id_id= '4526'
  and notice_id= '4526'
  and route_id= '4526';
select agency_id
from m.agency
where agency_id_id= '3030'
  and valid_now=13997;
select agency_id
from m.agency
where agency_id_id= '3184'
  and valid_now=3550;
select user_id
from m.agency
where valid_now=13459
  and agency_id_id= '6817';
select user_id
from m.agency
where valid_now=9333
  and agency_id_id= '2010';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8345'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=104
  and agency_id_id= '3623';
select COUNT(*)
from dv.notes_message
where user_id='7411'
  and agency_id_id= '7411'
  and notice_id= '7411'
  and route_id= '7411';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19827'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16279'
  and valid_now=12088;
select user_id
from m.agency
where valid_now=12703
  and agency_id_id= '4537';
select user_id
from m.agency
where valid_now=5209
  and agency_id_id= '15015';
select COUNT(*)
from dv.notes_message
where user_id='1916'
  and agency_id_id= '1916'
  and notice_id= '1916'
  and route_id= '1916';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6608'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14897'
  and valid_now=1541;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18119'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7959'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16211'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10464'
  and valid_now=16272;
select COUNT(*)
from dv.notes_message
where user_id='5605'
  and agency_id_id= '5605'
  and notice_id= '5605'
  and route_id= '5605';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10416'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2445'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3072'
  and agency_id_id= '3072'
  and notice_id= '3072'
  and route_id= '3072';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '925'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '150'
  and valid_now=17520;
select agency_id
from m.agency
where agency_id_id= '386'
  and valid_now=17398;
select user_id
from m.agency
where valid_now=6757
  and agency_id_id= '13652';
select user_id
from m.agency
where valid_now=19259
  and agency_id_id= '19983';
select COUNT(*)
from dv.notes_message
where user_id='9837'
  and agency_id_id= '9837'
  and notice_id= '9837'
  and route_id= '9837';
select COUNT(*)
from dv.notes_message
where user_id='8774'
  and agency_id_id= '8774'
  and notice_id= '8774'
  and route_id= '8774';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15824'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10900'
  and valid_now=351;
select user_id
from m.agency
where valid_now=5566
  and agency_id_id= '14644';
select user_id
from m.agency
where valid_now=3649
  and agency_id_id= '4659';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '928'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15285'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19448'
  and valid_now=6417;
select COUNT(*)
from dv.notes_message
where user_id='6108'
  and agency_id_id= '6108'
  and notice_id= '6108'
  and route_id= '6108';
select user_id
from m.agency
where valid_now=8269
  and agency_id_id= '15890';
select user_id
from m.agency
where valid_now=8765
  and agency_id_id= '2119';
select user_id
from m.agency
where valid_now=3363
  and agency_id_id= '3224';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12943'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19358'
  and valid_now=385;
select user_id
from m.agency
where valid_now=17577
  and agency_id_id= '15806';
select COUNT(*)
from dv.notes_message
where user_id='19019'
  and agency_id_id= '19019'
  and notice_id= '19019'
  and route_id= '19019';
select COUNT(*)
from dv.notes_message
where user_id='10263'
  and agency_id_id= '10263'
  and notice_id= '10263'
  and route_id= '10263';
select agency_id
from m.agency
where agency_id_id= '4312'
  and valid_now=19115;
select user_id
from m.agency
where valid_now=13412
  and agency_id_id= '8985';
select user_id
from m.agency
where valid_now=14872
  and agency_id_id= '12276';
select agency_id
from m.agency
where agency_id_id= '7004'
  and valid_now=16583;
select user_id
from m.agency
where valid_now=9953
  and agency_id_id= '15643';
select COUNT(*)
from dv.notes_message
where user_id='1975'
  and agency_id_id= '1975'
  and notice_id= '1975'
  and route_id= '1975';
select COUNT(*)
from dv.notes_message
where user_id='14281'
  and agency_id_id= '14281'
  and notice_id= '14281'
  and route_id= '14281';
select COUNT(*)
from dv.notes_message
where user_id='7928'
  and agency_id_id= '7928'
  and notice_id= '7928'
  and route_id= '7928';
select COUNT(*)
from dv.notes_message
where user_id='17050'
  and agency_id_id= '17050'
  and notice_id= '17050'
  and route_id= '17050';
select agency_id
from m.agency
where agency_id_id= '4058'
  and valid_now=13300;
select user_id
from m.agency
where valid_now=2354
  and agency_id_id= '18041';
select agency_id
from m.agency
where agency_id_id= '19582'
  and valid_now=11191;
select agency_id
from m.agency
where agency_id_id= '2206'
  and valid_now=8981;
select user_id
from m.agency
where valid_now=2471
  and agency_id_id= '9687';
select COUNT(*)
from dv.notes_message
where user_id='8845'
  and agency_id_id= '8845'
  and notice_id= '8845'
  and route_id= '8845';
select agency_id
from m.agency
where agency_id_id= '7251'
  and valid_now=14954;
select agency_id
from m.agency
where agency_id_id= '5646'
  and valid_now=1250;
select user_id
from m.agency
where valid_now=9593
  and agency_id_id= '7228';
select agency_id
from m.agency
where agency_id_id= '17757'
  and valid_now=10557;
select COUNT(*)
from dv.notes_message
where user_id='6769'
  and agency_id_id= '6769'
  and notice_id= '6769'
  and route_id= '6769';
select COUNT(*)
from dv.notes_message
where user_id='5283'
  and agency_id_id= '5283'
  and notice_id= '5283'
  and route_id= '5283';
select agency_id
from m.agency
where agency_id_id= '5905'
  and valid_now=7626;
select agency_id
from m.agency
where agency_id_id= '14540'
  and valid_now=16841;
select user_id
from m.agency
where valid_now=245
  and agency_id_id= '11594';
select COUNT(*)
from dv.notes_message
where user_id='10295'
  and agency_id_id= '10295'
  and notice_id= '10295'
  and route_id= '10295';
select user_id
from m.agency
where valid_now=12081
  and agency_id_id= '15063';
select agency_id
from m.agency
where agency_id_id= '3369'
  and valid_now=2932;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5538'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15800'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14291'
  and agency_id_id= '14291'
  and notice_id= '14291'
  and route_id= '14291';
select COUNT(*)
from dv.notes_message
where user_id='10971'
  and agency_id_id= '10971'
  and notice_id= '10971'
  and route_id= '10971';
select COUNT(*)
from dv.notes_message
where user_id='4711'
  and agency_id_id= '4711'
  and notice_id= '4711'
  and route_id= '4711';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19058'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2675'
  and valid_now=2505;
select user_id
from m.agency
where valid_now=18098
  and agency_id_id= '3973';
select COUNT(*)
from dv.notes_message
where user_id='8644'
  and agency_id_id= '8644'
  and notice_id= '8644'
  and route_id= '8644';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2724'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3914
  and agency_id_id= '5955';
select agency_id
from m.agency
where agency_id_id= '2316'
  and valid_now=11655;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14228'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18483'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18831
  and agency_id_id= '12108';
select COUNT(*)
from dv.notes_message
where user_id='6638'
  and agency_id_id= '6638'
  and notice_id= '6638'
  and route_id= '6638';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3548'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19606
  and agency_id_id= '566';
select COUNT(*)
from dv.notes_message
where user_id='17173'
  and agency_id_id= '17173'
  and notice_id= '17173'
  and route_id= '17173';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17116'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1217'
  and agency_id_id= '1217'
  and notice_id= '1217'
  and route_id= '1217';
select agency_id
from m.agency
where agency_id_id= '4607'
  and valid_now=7517;
select agency_id
from m.agency
where agency_id_id= '71'
  and valid_now=1811;
select user_id
from m.agency
where valid_now=3078
  and agency_id_id= '18388';
select COUNT(*)
from dv.notes_message
where user_id='2811'
  and agency_id_id= '2811'
  and notice_id= '2811'
  and route_id= '2811';
select COUNT(*)
from dv.notes_message
where user_id='12214'
  and agency_id_id= '12214'
  and notice_id= '12214'
  and route_id= '12214';
select agency_id
from m.agency
where agency_id_id= '12287'
  and valid_now=16721;
select user_id
from m.agency
where valid_now=7635
  and agency_id_id= '1271';
select COUNT(*)
from dv.notes_message
where user_id='628'
  and agency_id_id= '628'
  and notice_id= '628'
  and route_id= '628';
select agency_id
from m.agency
where agency_id_id= '11476'
  and valid_now=10868;
select user_id
from m.agency
where valid_now=10872
  and agency_id_id= '6588';
select agency_id
from m.agency
where agency_id_id= '18673'
  and valid_now=8344;
select user_id
from m.agency
where valid_now=12903
  and agency_id_id= '15325';
select user_id
from m.agency
where valid_now=9126
  and agency_id_id= '9602';
select user_id
from m.agency
where valid_now=589
  and agency_id_id= '11450';
select COUNT(*)
from dv.notes_message
where user_id='10098'
  and agency_id_id= '10098'
  and notice_id= '10098'
  and route_id= '10098';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4604'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11966'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14567'
  and valid_now=1928;
select user_id
from m.agency
where valid_now=15048
  and agency_id_id= '13512';
select COUNT(*)
from dv.notes_message
where user_id='7032'
  and agency_id_id= '7032'
  and notice_id= '7032'
  and route_id= '7032';
select COUNT(*)
from dv.notes_message
where user_id='3792'
  and agency_id_id= '3792'
  and notice_id= '3792'
  and route_id= '3792';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5789'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2013
  and agency_id_id= '5238';
select COUNT(*)
from dv.notes_message
where user_id='9127'
  and agency_id_id= '9127'
  and notice_id= '9127'
  and route_id= '9127';
select COUNT(*)
from dv.notes_message
where user_id='16547'
  and agency_id_id= '16547'
  and notice_id= '16547'
  and route_id= '16547';
select agency_id
from m.agency
where agency_id_id= '13204'
  and valid_now=16904;
select user_id
from m.agency
where valid_now=14258
  and agency_id_id= '7318';
select COUNT(*)
from dv.notes_message
where user_id='3886'
  and agency_id_id= '3886'
  and notice_id= '3886'
  and route_id= '3886';
select COUNT(*)
from dv.notes_message
where user_id='14293'
  and agency_id_id= '14293'
  and notice_id= '14293'
  and route_id= '14293';
select COUNT(*)
from dv.notes_message
where user_id='18907'
  and agency_id_id= '18907'
  and notice_id= '18907'
  and route_id= '18907';
select agency_id
from m.agency
where agency_id_id= '8750'
  and valid_now=7964;
select agency_id
from m.agency
where agency_id_id= '10559'
  and valid_now=13691;
select user_id
from m.agency
where valid_now=7283
  and agency_id_id= '531';
select COUNT(*)
from dv.notes_message
where user_id='12201'
  and agency_id_id= '12201'
  and notice_id= '12201'
  and route_id= '12201';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15965'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8266'
  and valid_now=3576;
select user_id
from m.agency
where valid_now=16451
  and agency_id_id= '18851';
select user_id
from m.agency
where valid_now=2777
  and agency_id_id= '16751';
select user_id
from m.agency
where valid_now=15500
  and agency_id_id= '9406';
select COUNT(*)
from dv.notes_message
where user_id='16086'
  and agency_id_id= '16086'
  and notice_id= '16086'
  and route_id= '16086';
select agency_id
from m.agency
where agency_id_id= '382'
  and valid_now=15789;
select agency_id
from m.agency
where agency_id_id= '19863'
  and valid_now=19075;
select agency_id
from m.agency
where agency_id_id= '379'
  and valid_now=13143;
select agency_id
from m.agency
where agency_id_id= '8749'
  and valid_now=5678;
select user_id
from m.agency
where valid_now=16776
  and agency_id_id= '13573';
select user_id
from m.agency
where valid_now=4173
  and agency_id_id= '12575';
select agency_id
from m.agency
where agency_id_id= '7464'
  and valid_now=3836;
select user_id
from m.agency
where valid_now=14367
  and agency_id_id= '4314';
select COUNT(*)
from dv.notes_message
where user_id='13998'
  and agency_id_id= '13998'
  and notice_id= '13998'
  and route_id= '13998';
select COUNT(*)
from dv.notes_message
where user_id='1876'
  and agency_id_id= '1876'
  and notice_id= '1876'
  and route_id= '1876';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4424'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12973'
  and valid_now=10162;
select user_id
from m.agency
where valid_now=10015
  and agency_id_id= '2432';
select user_id
from m.agency
where valid_now=18170
  and agency_id_id= '12969';
select user_id
from m.agency
where valid_now=4603
  and agency_id_id= '5476';
select user_id
from m.agency
where valid_now=13140
  and agency_id_id= '13026';
select COUNT(*)
from dv.notes_message
where user_id='17521'
  and agency_id_id= '17521'
  and notice_id= '17521'
  and route_id= '17521';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '901'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3900'
  and valid_now=404;
select user_id
from m.agency
where valid_now=15825
  and agency_id_id= '18603';
select COUNT(*)
from dv.notes_message
where user_id='9458'
  and agency_id_id= '9458'
  and notice_id= '9458'
  and route_id= '9458';
select COUNT(*)
from dv.notes_message
where user_id='527'
  and agency_id_id= '527'
  and notice_id= '527'
  and route_id= '527';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8936'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2889'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9157'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16235'
  and valid_now=12804;
select agency_id
from m.agency
where agency_id_id= '9674'
  and valid_now=2914;
select user_id
from m.agency
where valid_now=18264
  and agency_id_id= '4613';
select user_id
from m.agency
where valid_now=8588
  and agency_id_id= '5935';
select COUNT(*)
from dv.notes_message
where user_id='15814'
  and agency_id_id= '15814'
  and notice_id= '15814'
  and route_id= '15814';
select COUNT(*)
from dv.notes_message
where user_id='6404'
  and agency_id_id= '6404'
  and notice_id= '6404'
  and route_id= '6404';
select COUNT(*)
from dv.notes_message
where user_id='13207'
  and agency_id_id= '13207'
  and notice_id= '13207'
  and route_id= '13207';
select COUNT(*)
from dv.notes_message
where user_id='491'
  and agency_id_id= '491'
  and notice_id= '491'
  and route_id= '491';
select user_id
from m.agency
where valid_now=5585
  and agency_id_id= '12670';
select user_id
from m.agency
where valid_now=7474
  and agency_id_id= '16876';
select user_id
from m.agency
where valid_now=17852
  and agency_id_id= '16645';
select user_id
from m.agency
where valid_now=1219
  and agency_id_id= '2139';
select user_id
from m.agency
where valid_now=8708
  and agency_id_id= '7679';
select COUNT(*)
from dv.notes_message
where user_id='9989'
  and agency_id_id= '9989'
  and notice_id= '9989'
  and route_id= '9989';
select COUNT(*)
from dv.notes_message
where user_id='19398'
  and agency_id_id= '19398'
  and notice_id= '19398'
  and route_id= '19398';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5998'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9484'
  and valid_now=8693;
select user_id
from m.agency
where valid_now=14201
  and agency_id_id= '5211';
select agency_id
from m.agency
where agency_id_id= '9162'
  and valid_now=12925;
select user_id
from m.agency
where valid_now=4062
  and agency_id_id= '8638';
select COUNT(*)
from dv.notes_message
where user_id='19020'
  and agency_id_id= '19020'
  and notice_id= '19020'
  and route_id= '19020';
select agency_id
from m.agency
where agency_id_id= '7250'
  and valid_now=3198;
select COUNT(*)
from dv.notes_message
where user_id='16205'
  and agency_id_id= '16205'
  and notice_id= '16205'
  and route_id= '16205';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6319'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2722'
  and valid_now=18136;
select agency_id
from m.agency
where agency_id_id= '13567'
  and valid_now=10873;
select user_id
from m.agency
where valid_now=10971
  and agency_id_id= '17776';
select user_id
from m.agency
where valid_now=15187
  and agency_id_id= '15539';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11040'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16127'
  and valid_now=3081;
select COUNT(*)
from dv.notes_message
where user_id='19995'
  and agency_id_id= '19995'
  and notice_id= '19995'
  and route_id= '19995';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16057'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6293'
  and valid_now=3620;
select agency_id
from m.agency
where agency_id_id= '18829'
  and valid_now=13148;
select user_id
from m.agency
where valid_now=16787
  and agency_id_id= '1971';
select COUNT(*)
from dv.notes_message
where user_id='19834'
  and agency_id_id= '19834'
  and notice_id= '19834'
  and route_id= '19834';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11769'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5678'
  and valid_now=2507;
select user_id
from m.agency
where valid_now=1430
  and agency_id_id= '6989';
select COUNT(*)
from dv.notes_message
where user_id='13658'
  and agency_id_id= '13658'
  and notice_id= '13658'
  and route_id= '13658';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8562'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13871'
  and valid_now=6802;
select agency_id
from m.agency
where agency_id_id= '19058'
  and valid_now=3218;
select agency_id
from m.agency
where agency_id_id= '5318'
  and valid_now=2602;
select user_id
from m.agency
where valid_now=6128
  and agency_id_id= '11018';
select COUNT(*)
from dv.notes_message
where user_id='15148'
  and agency_id_id= '15148'
  and notice_id= '15148'
  and route_id= '15148';
select COUNT(*)
from dv.notes_message
where user_id='12912'
  and agency_id_id= '12912'
  and notice_id= '12912'
  and route_id= '12912';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3647'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8745'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8541'
  and valid_now=11490;
select user_id
from m.agency
where valid_now=9179
  and agency_id_id= '2395';
select user_id
from m.agency
where valid_now=6925
  and agency_id_id= '16660';
select agency_id
from m.agency
where agency_id_id= '19237'
  and valid_now=8470;
select COUNT(*)
from dv.notes_message
where user_id='11630'
  and agency_id_id= '11630'
  and notice_id= '11630'
  and route_id= '11630';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6168'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4237'
  and valid_now=12447;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12441'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7612
  and agency_id_id= '4088';
select user_id
from m.agency
where valid_now=14521
  and agency_id_id= '19779';
select user_id
from m.agency
where valid_now=18711
  and agency_id_id= '9681';
select COUNT(*)
from dv.notes_message
where user_id='5298'
  and agency_id_id= '5298'
  and notice_id= '5298'
  and route_id= '5298';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16405'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12733'
  and valid_now=3480;
select agency_id
from m.agency
where agency_id_id= '9361'
  and valid_now=18972;
select user_id
from m.agency
where valid_now=3414
  and agency_id_id= '18149';
select agency_id
from m.agency
where agency_id_id= '12693'
  and valid_now=9706;
select user_id
from m.agency
where valid_now=16193
  and agency_id_id= '13311';
select user_id
from m.agency
where valid_now=8498
  and agency_id_id= '17835';
select user_id
from m.agency
where valid_now=115
  and agency_id_id= '16743';
select COUNT(*)
from dv.notes_message
where user_id='14447'
  and agency_id_id= '14447'
  and notice_id= '14447'
  and route_id= '14447';
select agency_id
from m.agency
where agency_id_id= '13320'
  and valid_now=4455;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3761'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16913'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5787
  and agency_id_id= '7946';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10540'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14673'
  and valid_now=19585;
select COUNT(*)
from dv.notes_message
where user_id='15858'
  and agency_id_id= '15858'
  and notice_id= '15858'
  and route_id= '15858';
select COUNT(*)
from dv.notes_message
where user_id='10320'
  and agency_id_id= '10320'
  and notice_id= '10320'
  and route_id= '10320';
select COUNT(*)
from dv.notes_message
where user_id='13087'
  and agency_id_id= '13087'
  and notice_id= '13087'
  and route_id= '13087';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13778'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '10260';
select agency_id
from m.agency
where agency_id_id= '8139'
  and valid_now=19392;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9147'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5041';
select a.agency_timezone
from m.agency a
where a.agency_id = '16504';
select a.agency_timezone
from m.agency a
where a.agency_id = '376';
select agency_id
from m.agency
where agency_id_id= '13756'
  and valid_now=6396;
select COUNT(*)
from dv.notes_message
where user_id='4085'
  and agency_id_id= '4085'
  and notice_id= '4085'
  and route_id= '4085';
select a.agency_timezone
from m.agency a
where a.agency_id = '3158';
select agency_id
from m.agency
where agency_id_id= '6830'
  and valid_now=4932;
select agency_id
from m.agency
where agency_id_id= '3286'
  and valid_now=7719;
select agency_id
from m.agency
where agency_id_id= '12678'
  and valid_now=5462;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9276'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16378'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '15785';
select agency_id
from m.agency
where agency_id_id= '14095'
  and valid_now=8562;
select agency_id
from m.agency
where agency_id_id= '10293'
  and valid_now=12481;
select a.agency_timezone
from m.agency a
where a.agency_id = '9303';
select COUNT(*)
from dv.notes_message
where user_id='2312'
  and agency_id_id= '2312'
  and notice_id= '2312'
  and route_id= '2312';
select a.agency_timezone
from m.agency a
where a.agency_id = '17652';
select a.agency_timezone
from m.agency a
where a.agency_id = '7598';
select a.agency_timezone
from m.agency a
where a.agency_id = '18689';
select COUNT(*)
from dv.notes_message
where user_id='2610'
  and agency_id_id= '2610'
  and notice_id= '2610'
  and route_id= '2610';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17811'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1386'
  and valid_now=10368;
select user_id
from m.agency
where valid_now=5899
  and agency_id_id= '11374';
select user_id
from m.agency
where valid_now=5431
  and agency_id_id= '4117';
select COUNT(*)
from dv.notes_message
where user_id='1194'
  and agency_id_id= '1194'
  and notice_id= '1194'
  and route_id= '1194';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5542'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15221'
  and valid_now=5685;
select agency_id
from m.agency
where agency_id_id= '5119'
  and valid_now=4767;
select user_id
from m.agency
where valid_now=9244
  and agency_id_id= '12081';
select COUNT(*)
from dv.notes_message
where user_id='5185'
  and agency_id_id= '5185'
  and notice_id= '5185'
  and route_id= '5185';
select COUNT(*)
from dv.notes_message
where user_id='8931'
  and agency_id_id= '8931'
  and notice_id= '8931'
  and route_id= '8931';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18861'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19328'
  and valid_now=774;
select user_id
from m.agency
where valid_now=3727
  and agency_id_id= '13039';
select agency_id
from m.agency
where agency_id_id= '11225'
  and valid_now=16096;
select agency_id
from m.agency
where agency_id_id= '3497'
  and valid_now=4523;
select agency_id
from m.agency
where agency_id_id= '14313'
  and valid_now=12717;
select COUNT(*)
from dv.notes_message
where user_id='6416'
  and agency_id_id= '6416'
  and notice_id= '6416'
  and route_id= '6416';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9492'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8880'
  and agency_id_id= '8880'
  and notice_id= '8880'
  and route_id= '8880';
select a.agency_timezone
from m.agency a
where a.agency_id = '16848';
select user_id
from m.agency
where valid_now=4623
  and agency_id_id= '9256';
select user_id
from m.agency
where valid_now=7078
  and agency_id_id= '8712';
select COUNT(*)
from dv.notes_message
where user_id='12437'
  and agency_id_id= '12437'
  and notice_id= '12437'
  and route_id= '12437';
select agency_id
from m.agency
where agency_id_id= '14012'
  and valid_now=1571;
select agency_id
from m.agency
where agency_id_id= '11728'
  and valid_now=4340;
select user_id
from m.agency
where valid_now=10245
  and agency_id_id= '12906';
select user_id
from m.agency
where valid_now=17073
  and agency_id_id= '4015';
select COUNT(*)
from dv.notes_message
where user_id='10967'
  and agency_id_id= '10967'
  and notice_id= '10967'
  and route_id= '10967';
select agency_id
from m.agency
where agency_id_id= '13888'
  and valid_now=12834;
select agency_id
from m.agency
where agency_id_id= '16410'
  and valid_now=8682;
select user_id
from m.agency
where valid_now=18329
  and agency_id_id= '9992';
select COUNT(*)
from dv.notes_message
where user_id='3167'
  and agency_id_id= '3167'
  and notice_id= '3167'
  and route_id= '3167';
select COUNT(*)
from dv.notes_message
where user_id='8075'
  and agency_id_id= '8075'
  and notice_id= '8075'
  and route_id= '8075';
select a.agency_timezone
from m.agency a
where a.agency_id = '4129';
select user_id
from m.agency
where valid_now=15502
  and agency_id_id= '11638';
select a.agency_timezone
from m.agency a
where a.agency_id = '18871';
select user_id
from m.agency
where valid_now=18761
  and agency_id_id= '18891';
select COUNT(*)
from dv.notes_message
where user_id='16454'
  and agency_id_id= '16454'
  and notice_id= '16454'
  and route_id= '16454';
select COUNT(*)
from dv.notes_message
where user_id='10630'
  and agency_id_id= '10630'
  and notice_id= '10630'
  and route_id= '10630';
select COUNT(*)
from dv.notes_message
where user_id='4083'
  and agency_id_id= '4083'
  and notice_id= '4083'
  and route_id= '4083';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14854'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18962'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15034'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11450'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16206'
  and valid_now=9884;
select agency_id
from m.agency
where agency_id_id= '13732'
  and valid_now=16043;
select user_id
from m.agency
where valid_now=2420
  and agency_id_id= '3560';
select COUNT(*)
from dv.notes_message
where user_id='13455'
  and agency_id_id= '13455'
  and notice_id= '13455'
  and route_id= '13455';
select COUNT(*)
from dv.notes_message
where user_id='7144'
  and agency_id_id= '7144'
  and notice_id= '7144'
  and route_id= '7144';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16250'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12449
  and agency_id_id= '18004';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9402'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16299'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8613'
  and valid_now=19052;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8507'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11776'
  and agency_id_id= '11776'
  and notice_id= '11776'
  and route_id= '11776';
select COUNT(*)
from dv.notes_message
where user_id='15825'
  and agency_id_id= '15825'
  and notice_id= '15825'
  and route_id= '15825';
select user_id
from m.agency
where valid_now=11481
  and agency_id_id= '7480';
select user_id
from m.agency
where valid_now=2030
  and agency_id_id= '4676';
select user_id
from m.agency
where valid_now=11786
  and agency_id_id= '3504';
select user_id
from m.agency
where valid_now=9924
  and agency_id_id= '1033';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6464'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9864'
  and valid_now=279;
select agency_id
from m.agency
where agency_id_id= '3348'
  and valid_now=2058;
select user_id
from m.agency
where valid_now=7372
  and agency_id_id= '5751';
select COUNT(*)
from dv.notes_message
where user_id='3532'
  and agency_id_id= '3532'
  and notice_id= '3532'
  and route_id= '3532';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5859'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '34'
  and valid_now=18582;
select agency_id
from m.agency
where agency_id_id= '16660'
  and valid_now=15607;
select user_id
from m.agency
where valid_now=8728
  and agency_id_id= '10509';
select COUNT(*)
from dv.notes_message
where user_id='878'
  and agency_id_id= '878'
  and notice_id= '878'
  and route_id= '878';
select COUNT(*)
from dv.notes_message
where user_id='5319'
  and agency_id_id= '5319'
  and notice_id= '5319'
  and route_id= '5319';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15835'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1176'
  and valid_now=9293;
select agency_id
from m.agency
where agency_id_id= '5862'
  and valid_now=19217;
select user_id
from m.agency
where valid_now=17672
  and agency_id_id= '17586';
select user_id
from m.agency
where valid_now=1162
  and agency_id_id= '10441';
select user_id
from m.agency
where valid_now=6360
  and agency_id_id= '3866';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7754'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2228'
  and valid_now=13671;
select agency_id
from m.agency
where agency_id_id= '283'
  and valid_now=19682;
select user_id
from m.agency
where valid_now=1471
  and agency_id_id= '19664';
select user_id
from m.agency
where valid_now=13756
  and agency_id_id= '7043';
select COUNT(*)
from dv.notes_message
where user_id='2675'
  and agency_id_id= '2675'
  and notice_id= '2675'
  and route_id= '2675';
select agency_id
from m.agency
where agency_id_id= '4001'
  and valid_now=16133;
select agency_id
from m.agency
where agency_id_id= '11443'
  and valid_now=19708;
select agency_id
from m.agency
where agency_id_id= '17456'
  and valid_now=6066;
select agency_id
from m.agency
where agency_id_id= '12423'
  and valid_now=4898;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19286'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6124
  and agency_id_id= '2340';
select COUNT(*)
from dv.notes_message
where user_id='6424'
  and agency_id_id= '6424'
  and notice_id= '6424'
  and route_id= '6424';
select agency_id
from m.agency
where agency_id_id= '2070'
  and valid_now=15556;
select agency_id
from m.agency
where agency_id_id= '19229'
  and valid_now=10029;
select COUNT(*)
from dv.notes_message
where user_id='1818'
  and agency_id_id= '1818'
  and notice_id= '1818'
  and route_id= '1818';
select COUNT(*)
from dv.notes_message
where user_id='7952'
  and agency_id_id= '7952'
  and notice_id= '7952'
  and route_id= '7952';
select COUNT(*)
from dv.notes_message
where user_id='15470'
  and agency_id_id= '15470'
  and notice_id= '15470'
  and route_id= '15470';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11903'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6050'
  and valid_now=17530;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8002'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10146'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5375'
  and valid_now=6129;
select agency_id
from m.agency
where agency_id_id= '15408'
  and valid_now=14828;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1808'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4370'
  and valid_now=14731;
select user_id
from m.agency
where valid_now=12460
  and agency_id_id= '9952';
select COUNT(*)
from dv.notes_message
where user_id='9243'
  and agency_id_id= '9243'
  and notice_id= '9243'
  and route_id= '9243';
select agency_id
from m.agency
where agency_id_id= '1500'
  and valid_now=14110;
select user_id
from m.agency
where valid_now=16505
  and agency_id_id= '17340';
select user_id
from m.agency
where valid_now=4009
  and agency_id_id= '13609';
select COUNT(*)
from dv.notes_message
where user_id='7216'
  and agency_id_id= '7216'
  and notice_id= '7216'
  and route_id= '7216';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8093'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8410
  and agency_id_id= '13285';
select COUNT(*)
from dv.notes_message
where user_id='1878'
  and agency_id_id= '1878'
  and notice_id= '1878'
  and route_id= '1878';
select COUNT(*)
from dv.notes_message
where user_id='1538'
  and agency_id_id= '1538'
  and notice_id= '1538'
  and route_id= '1538';
select COUNT(*)
from dv.notes_message
where user_id='18684'
  and agency_id_id= '18684'
  and notice_id= '18684'
  and route_id= '18684';
select user_id
from m.agency
where valid_now=2642
  and agency_id_id= '17584';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13109'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11070'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8590'
  and valid_now=14741;
select agency_id
from m.agency
where agency_id_id= '16378'
  and valid_now=5355;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12997'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3226'
  and valid_now=19195;
select COUNT(*)
from dv.notes_message
where user_id='13005'
  and agency_id_id= '13005'
  and notice_id= '13005'
  and route_id= '13005';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16233'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3969'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14170'
  and valid_now=8481;
select user_id
from m.agency
where valid_now=16276
  and agency_id_id= '19556';
select COUNT(*)
from dv.notes_message
where user_id='2461'
  and agency_id_id= '2461'
  and notice_id= '2461'
  and route_id= '2461';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1921'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8143'
  and valid_now=13692;
select user_id
from m.agency
where valid_now=19685
  and agency_id_id= '9565';
select user_id
from m.agency
where valid_now=8269
  and agency_id_id= '17641';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14102'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7077'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11041
  and agency_id_id= '11927';
select agency_id
from m.agency
where agency_id_id= '16750'
  and valid_now=15231;
select user_id
from m.agency
where valid_now=13787
  and agency_id_id= '12979';
select user_id
from m.agency
where valid_now=4971
  and agency_id_id= '14845';
select agency_id
from m.agency
where agency_id_id= '4043'
  and valid_now=1262;
select user_id
from m.agency
where valid_now=2931
  and agency_id_id= '6113';
select user_id
from m.agency
where valid_now=18863
  and agency_id_id= '855';
select COUNT(*)
from dv.notes_message
where user_id='15129'
  and agency_id_id= '15129'
  and notice_id= '15129'
  and route_id= '15129';
select COUNT(*)
from dv.notes_message
where user_id='11761'
  and agency_id_id= '11761'
  and notice_id= '11761'
  and route_id= '11761';
select COUNT(*)
from dv.notes_message
where user_id='10761'
  and agency_id_id= '10761'
  and notice_id= '10761'
  and route_id= '10761';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12622'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18706'
  and valid_now=300;
select agency_id
from m.agency
where agency_id_id= '4221'
  and valid_now=2503;
select user_id
from m.agency
where valid_now=7846
  and agency_id_id= '11980';
select COUNT(*)
from dv.notes_message
where user_id='9060'
  and agency_id_id= '9060'
  and notice_id= '9060'
  and route_id= '9060';
select COUNT(*)
from dv.notes_message
where user_id='7984'
  and agency_id_id= '7984'
  and notice_id= '7984'
  and route_id= '7984';
select COUNT(*)
from dv.notes_message
where user_id='15001'
  and agency_id_id= '15001'
  and notice_id= '15001'
  and route_id= '15001';
select user_id
from m.agency
where valid_now=901
  and agency_id_id= '11376';
select COUNT(*)
from dv.notes_message
where user_id='5823'
  and agency_id_id= '5823'
  and notice_id= '5823'
  and route_id= '5823';
select COUNT(*)
from dv.notes_message
where user_id='19773'
  and agency_id_id= '19773'
  and notice_id= '19773'
  and route_id= '19773';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3572'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5539'
  and valid_now=5916;
select user_id
from m.agency
where valid_now=5361
  and agency_id_id= '2041';
select COUNT(*)
from dv.notes_message
where user_id='6507'
  and agency_id_id= '6507'
  and notice_id= '6507'
  and route_id= '6507';
select COUNT(*)
from dv.notes_message
where user_id='3779'
  and agency_id_id= '3779'
  and notice_id= '3779'
  and route_id= '3779';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5253'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '752'
  and valid_now=12308;
select user_id
from m.agency
where valid_now=17513
  and agency_id_id= '2830';
select COUNT(*)
from dv.notes_message
where user_id='305'
  and agency_id_id= '305'
  and notice_id= '305'
  and route_id= '305';
select COUNT(*)
from dv.notes_message
where user_id='19733'
  and agency_id_id= '19733'
  and notice_id= '19733'
  and route_id= '19733';
select agency_id
from m.agency
where agency_id_id= '15103'
  and valid_now=4945;
select agency_id
from m.agency
where agency_id_id= '9815'
  and valid_now=1109;
select agency_id
from m.agency
where agency_id_id= '11253'
  and valid_now=18993;
select user_id
from m.agency
where valid_now=1475
  and agency_id_id= '2731';
select user_id
from m.agency
where valid_now=4033
  and agency_id_id= '11541';
select COUNT(*)
from dv.notes_message
where user_id='14301'
  and agency_id_id= '14301'
  and notice_id= '14301'
  and route_id= '14301';
select agency_id
from m.agency
where agency_id_id= '14850'
  and valid_now=14859;
select agency_id
from m.agency
where agency_id_id= '17896'
  and valid_now=14588;
select agency_id
from m.agency
where agency_id_id= '2838'
  and valid_now=12582;
select user_id
from m.agency
where valid_now=16439
  and agency_id_id= '13710';
select COUNT(*)
from dv.notes_message
where user_id='8590'
  and agency_id_id= '8590'
  and notice_id= '8590'
  and route_id= '8590';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11554'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3625'
  and valid_now=4974;
select COUNT(*)
from dv.notes_message
where user_id='3722'
  and agency_id_id= '3722'
  and notice_id= '3722'
  and route_id= '3722';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14128'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12236'
  and agency_id_id= '12236'
  and notice_id= '12236'
  and route_id= '12236';
select COUNT(*)
from dv.notes_message
where user_id='133'
  and agency_id_id= '133'
  and notice_id= '133'
  and route_id= '133';
select COUNT(*)
from dv.notes_message
where user_id='18765'
  and agency_id_id= '18765'
  and notice_id= '18765'
  and route_id= '18765';
select COUNT(*)
from dv.notes_message
where user_id='18544'
  and agency_id_id= '18544'
  and notice_id= '18544'
  and route_id= '18544';
select user_id
from m.agency
where valid_now=8507
  and agency_id_id= '15703';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15105'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3417'
  and valid_now=15810;
select agency_id
from m.agency
where agency_id_id= '14157'
  and valid_now=13430;
select agency_id
from m.agency
where agency_id_id= '17740'
  and valid_now=17414;
select user_id
from m.agency
where valid_now=8891
  and agency_id_id= '6077';
select user_id
from m.agency
where valid_now=11561
  and agency_id_id= '3564';
select agency_id
from m.agency
where agency_id_id= '18687'
  and valid_now=18651;
select agency_id
from m.agency
where agency_id_id= '4502'
  and valid_now=9500;
select user_id
from m.agency
where valid_now=11466
  and agency_id_id= '19816';
select user_id
from m.agency
where valid_now=6886
  and agency_id_id= '19549';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8207'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10033'
  and valid_now=7232;
select user_id
from m.agency
where valid_now=18289
  and agency_id_id= '6664';
select COUNT(*)
from dv.notes_message
where user_id='12287'
  and agency_id_id= '12287'
  and notice_id= '12287'
  and route_id= '12287';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14082'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7599'
  and valid_now=17985;
select user_id
from m.agency
where valid_now=15819
  and agency_id_id= '11918';
select user_id
from m.agency
where valid_now=19047
  and agency_id_id= '13080';
select COUNT(*)
from dv.notes_message
where user_id='7127'
  and agency_id_id= '7127'
  and notice_id= '7127'
  and route_id= '7127';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7774'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16464'
  and valid_now=4293;
select user_id
from m.agency
where valid_now=12422
  and agency_id_id= '5234';
select user_id
from m.agency
where valid_now=3422
  and agency_id_id= '1654';
select agency_id
from m.agency
where agency_id_id= '12528'
  and valid_now=4769;
select user_id
from m.agency
where valid_now=9822
  and agency_id_id= '19429';
select COUNT(*)
from dv.notes_message
where user_id='18454'
  and agency_id_id= '18454'
  and notice_id= '18454'
  and route_id= '18454';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5337'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7836'
  and valid_now=5023;
select COUNT(*)
from dv.notes_message
where user_id='10061'
  and agency_id_id= '10061'
  and notice_id= '10061'
  and route_id= '10061';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2237'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18068'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '325'
  and valid_now=13144;
select COUNT(*)
from dv.notes_message
where user_id='18643'
  and agency_id_id= '18643'
  and notice_id= '18643'
  and route_id= '18643';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14135'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7232'
  and agency_id_id= '7232'
  and notice_id= '7232'
  and route_id= '7232';
select user_id
from m.agency
where valid_now=3010
  and agency_id_id= '18264';
select user_id
from m.agency
where valid_now=9366
  and agency_id_id= '1052';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13814'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3323'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14793'
  and valid_now=8470;
select agency_id
from m.agency
where agency_id_id= '1935'
  and valid_now=3071;
select COUNT(*)
from dv.notes_message
where user_id='6428'
  and agency_id_id= '6428'
  and notice_id= '6428'
  and route_id= '6428';
select COUNT(*)
from dv.notes_message
where user_id='10415'
  and agency_id_id= '10415'
  and notice_id= '10415'
  and route_id= '10415';
select agency_id
from m.agency
where agency_id_id= '16337'
  and valid_now=1835;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1389'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15710
  and agency_id_id= '13578';
select COUNT(*)
from dv.notes_message
where user_id='10509'
  and agency_id_id= '10509'
  and notice_id= '10509'
  and route_id= '10509';
select agency_id
from m.agency
where agency_id_id= '3558'
  and valid_now=13158;
select agency_id
from m.agency
where agency_id_id= '13673'
  and valid_now=15422;
select COUNT(*)
from dv.notes_message
where user_id='452'
  and agency_id_id= '452'
  and notice_id= '452'
  and route_id= '452';
select user_id
from m.agency
where valid_now=13537
  and agency_id_id= '9801';
select a.agency_timezone
from m.agency a
where a.agency_id = '492';
select agency_id
from m.agency
where agency_id_id= '15297'
  and valid_now=5038;
select agency_id
from m.agency
where agency_id_id= '3049'
  and valid_now=1105;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10319'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5245'
  and valid_now=2614;
select agency_id
from m.agency
where agency_id_id= '5102'
  and valid_now=2631;
select COUNT(*)
from dv.notes_message
where user_id='11317'
  and agency_id_id= '11317'
  and notice_id= '11317'
  and route_id= '11317';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19732'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6537'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9634
  and agency_id_id= '14180';
select agency_id
from m.agency
where agency_id_id= '19271'
  and valid_now=15408;
select agency_id
from m.agency
where agency_id_id= '14586'
  and valid_now=2089;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19681'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15648'
  and valid_now=19665;
select agency_id
from m.agency
where agency_id_id= '6188'
  and valid_now=14217;
select user_id
from m.agency
where valid_now=19615
  and agency_id_id= '824';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13512'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16149'
  and valid_now=14853;
select agency_id
from m.agency
where agency_id_id= '12408'
  and valid_now=933;
select agency_id
from m.agency
where agency_id_id= '19075'
  and valid_now=4647;
select user_id
from m.agency
where valid_now=11513
  and agency_id_id= '7327';
select user_id
from m.agency
where valid_now=12604
  and agency_id_id= '6409';
select agency_id
from m.agency
where agency_id_id= '3244'
  and valid_now=6120;
select user_id
from m.agency
where valid_now=10854
  and agency_id_id= '159';
select user_id
from m.agency
where valid_now=13426
  and agency_id_id= '650';
select user_id
from m.agency
where valid_now=5914
  and agency_id_id= '11313';
select user_id
from m.agency
where valid_now=5783
  and agency_id_id= '9343';
select agency_id
from m.agency
where agency_id_id= '18183'
  and valid_now=6129;
select user_id
from m.agency
where valid_now=8692
  and agency_id_id= '7300';
select user_id
from m.agency
where valid_now=6685
  and agency_id_id= '5684';
select COUNT(*)
from dv.notes_message
where user_id='17683'
  and agency_id_id= '17683'
  and notice_id= '17683'
  and route_id= '17683';
select agency_id
from m.agency
where agency_id_id= '9447'
  and valid_now=4453;
select COUNT(*)
from dv.notes_message
where user_id='4740'
  and agency_id_id= '4740'
  and notice_id= '4740'
  and route_id= '4740';
select COUNT(*)
from dv.notes_message
where user_id='8217'
  and agency_id_id= '8217'
  and notice_id= '8217'
  and route_id= '8217';
select COUNT(*)
from dv.notes_message
where user_id='1445'
  and agency_id_id= '1445'
  and notice_id= '1445'
  and route_id= '1445';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17230'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18811'
  and agency_id_id= '18811'
  and notice_id= '18811'
  and route_id= '18811';
select COUNT(*)
from dv.notes_message
where user_id='17653'
  and agency_id_id= '17653'
  and notice_id= '17653'
  and route_id= '17653';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1790'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '455'
  and valid_now=18942;
select user_id
from m.agency
where valid_now=3397
  and agency_id_id= '14151';
select COUNT(*)
from dv.notes_message
where user_id='14666'
  and agency_id_id= '14666'
  and notice_id= '14666'
  and route_id= '14666';
select user_id
from m.agency
where valid_now=12735
  and agency_id_id= '5598';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13204'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14800'
  and valid_now=5431;
select COUNT(*)
from dv.notes_message
where user_id='6773'
  and agency_id_id= '6773'
  and notice_id= '6773'
  and route_id= '6773';
select COUNT(*)
from dv.notes_message
where user_id='5327'
  and agency_id_id= '5327'
  and notice_id= '5327'
  and route_id= '5327';
select agency_id
from m.agency
where agency_id_id= '11548'
  and valid_now=7261;
select agency_id
from m.agency
where agency_id_id= '5619'
  and valid_now=714;
select user_id
from m.agency
where valid_now=15005
  and agency_id_id= '6407';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8249'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9702'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '527'
  and valid_now=7038;
select COUNT(*)
from dv.notes_message
where user_id='12142'
  and agency_id_id= '12142'
  and notice_id= '12142'
  and route_id= '12142';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14422'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13187'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8343
  and agency_id_id= '14250';
select user_id
from m.agency
where valid_now=19515
  and agency_id_id= '11944';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16851'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7493
  and agency_id_id= '14461';
select COUNT(*)
from dv.notes_message
where user_id='2590'
  and agency_id_id= '2590'
  and notice_id= '2590'
  and route_id= '2590';
select COUNT(*)
from dv.notes_message
where user_id='2692'
  and agency_id_id= '2692'
  and notice_id= '2692'
  and route_id= '2692';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12732'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9246'
  and valid_now=9070;
select user_id
from m.agency
where valid_now=9124
  and agency_id_id= '15617';
select user_id
from m.agency
where valid_now=2143
  and agency_id_id= '5060';
select COUNT(*)
from dv.notes_message
where user_id='5453'
  and agency_id_id= '5453'
  and notice_id= '5453'
  and route_id= '5453';
select COUNT(*)
from dv.notes_message
where user_id='19593'
  and agency_id_id= '19593'
  and notice_id= '19593'
  and route_id= '19593';
select agency_id
from m.agency
where agency_id_id= '16986'
  and valid_now=13951;
select user_id
from m.agency
where valid_now=11013
  and agency_id_id= '9549';
select user_id
from m.agency
where valid_now=4817
  and agency_id_id= '2892';
select COUNT(*)
from dv.notes_message
where user_id='15257'
  and agency_id_id= '15257'
  and notice_id= '15257'
  and route_id= '15257';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17776'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17923'
  and valid_now=4932;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9848'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15439'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2154'
  and valid_now=1348;
select agency_id
from m.agency
where agency_id_id= '10418'
  and valid_now=6111;
select agency_id
from m.agency
where agency_id_id= '3511'
  and valid_now=3647;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4673'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3592'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11789
  and agency_id_id= '10579';
select COUNT(*)
from dv.notes_message
where user_id='8244'
  and agency_id_id= '8244'
  and notice_id= '8244'
  and route_id= '8244';
select COUNT(*)
from dv.notes_message
where user_id='11050'
  and agency_id_id= '11050'
  and notice_id= '11050'
  and route_id= '11050';
select agency_id
from m.agency
where agency_id_id= '8204'
  and valid_now=16889;
select user_id
from m.agency
where valid_now=1187
  and agency_id_id= '10192';
select agency_id
from m.agency
where agency_id_id= '2875'
  and valid_now=7766;
select COUNT(*)
from dv.notes_message
where user_id='782'
  and agency_id_id= '782'
  and notice_id= '782'
  and route_id= '782';
select COUNT(*)
from dv.notes_message
where user_id='2779'
  and agency_id_id= '2779'
  and notice_id= '2779'
  and route_id= '2779';
select a.agency_timezone
from m.agency a
where a.agency_id = '16517';
select user_id
from m.agency
where valid_now=4638
  and agency_id_id= '11933';
select COUNT(*)
from dv.notes_message
where user_id='2684'
  and agency_id_id= '2684'
  and notice_id= '2684'
  and route_id= '2684';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2256'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17564'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18730
  and agency_id_id= '703';
select user_id
from m.agency
where valid_now=19528
  and agency_id_id= '1002';
select COUNT(*)
from dv.notes_message
where user_id='12224'
  and agency_id_id= '12224'
  and notice_id= '12224'
  and route_id= '12224';
select user_id
from m.agency
where valid_now=3836
  and agency_id_id= '1791';
select user_id
from m.agency
where valid_now=2070
  and agency_id_id= '17660';
select COUNT(*)
from dv.notes_message
where user_id='14107'
  and agency_id_id= '14107'
  and notice_id= '14107'
  and route_id= '14107';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13611'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4124'
  and valid_now=11446;
select user_id
from m.agency
where valid_now=7904
  and agency_id_id= '8807';
select agency_id
from m.agency
where agency_id_id= '4828'
  and valid_now=8793;
select agency_id
from m.agency
where agency_id_id= '3934'
  and valid_now=13689;
select COUNT(*)
from dv.notes_message
where user_id='5764'
  and agency_id_id= '5764'
  and notice_id= '5764'
  and route_id= '5764';
select COUNT(*)
from dv.notes_message
where user_id='5654'
  and agency_id_id= '5654'
  and notice_id= '5654'
  and route_id= '5654';
select agency_id
from m.agency
where agency_id_id= '3728'
  and valid_now=19376;
select user_id
from m.agency
where valid_now=13847
  and agency_id_id= '7490';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12354'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14839'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14367'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14238
  and agency_id_id= '11687';
select COUNT(*)
from dv.notes_message
where user_id='3747'
  and agency_id_id= '3747'
  and notice_id= '3747'
  and route_id= '3747';
select user_id
from m.agency
where valid_now=9604
  and agency_id_id= '11060';
select user_id
from m.agency
where valid_now=1714
  and agency_id_id= '13491';
select user_id
from m.agency
where valid_now=1438
  and agency_id_id= '11855';
select user_id
from m.agency
where valid_now=13526
  and agency_id_id= '9758';
select user_id
from m.agency
where valid_now=17433
  and agency_id_id= '14674';
select COUNT(*)
from dv.notes_message
where user_id='9260'
  and agency_id_id= '9260'
  and notice_id= '9260'
  and route_id= '9260';
select agency_id
from m.agency
where agency_id_id= '903'
  and valid_now=17923;
select agency_id
from m.agency
where agency_id_id= '6001'
  and valid_now=5098;
select COUNT(*)
from dv.notes_message
where user_id='15794'
  and agency_id_id= '15794'
  and notice_id= '15794'
  and route_id= '15794';
select COUNT(*)
from dv.notes_message
where user_id='3739'
  and agency_id_id= '3739'
  and notice_id= '3739'
  and route_id= '3739';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17999'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12784'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1076
  and agency_id_id= '9231';
select user_id
from m.agency
where valid_now=3333
  and agency_id_id= '18794';
select COUNT(*)
from dv.notes_message
where user_id='19347'
  and agency_id_id= '19347'
  and notice_id= '19347'
  and route_id= '19347';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12732'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12598
  and agency_id_id= '6041';
select user_id
from m.agency
where valid_now=16670
  and agency_id_id= '10729';
select user_id
from m.agency
where valid_now=7811
  and agency_id_id= '14965';
select COUNT(*)
from dv.notes_message
where user_id='3595'
  and agency_id_id= '3595'
  and notice_id= '3595'
  and route_id= '3595';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13695'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17986'
  and agency_id_id= '17986'
  and notice_id= '17986'
  and route_id= '17986';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8081'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7022'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12796'
  and valid_now=2266;
select agency_id
from m.agency
where agency_id_id= '13247'
  and valid_now=19837;
select user_id
from m.agency
where valid_now=5891
  and agency_id_id= '18946';
select user_id
from m.agency
where valid_now=3833
  and agency_id_id= '5129';
select COUNT(*)
from dv.notes_message
where user_id='173'
  and agency_id_id= '173'
  and notice_id= '173'
  and route_id= '173';
select COUNT(*)
from dv.notes_message
where user_id='16241'
  and agency_id_id= '16241'
  and notice_id= '16241'
  and route_id= '16241';
select COUNT(*)
from dv.notes_message
where user_id='11254'
  and agency_id_id= '11254'
  and notice_id= '11254'
  and route_id= '11254';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13961'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17146'
  and valid_now=10456;
select user_id
from m.agency
where valid_now=19981
  and agency_id_id= '13947';
select COUNT(*)
from dv.notes_message
where user_id='6874'
  and agency_id_id= '6874'
  and notice_id= '6874'
  and route_id= '6874';
select COUNT(*)
from dv.notes_message
where user_id='9006'
  and agency_id_id= '9006'
  and notice_id= '9006'
  and route_id= '9006';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1932'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2664'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8546'
  and agency_id_id= '8546'
  and notice_id= '8546'
  and route_id= '8546';
select COUNT(*)
from dv.notes_message
where user_id='4515'
  and agency_id_id= '4515'
  and notice_id= '4515'
  and route_id= '4515';
select COUNT(*)
from dv.notes_message
where user_id='13572'
  and agency_id_id= '13572'
  and notice_id= '13572'
  and route_id= '13572';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10176'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18718'
  and valid_now=5069;
select user_id
from m.agency
where valid_now=3080
  and agency_id_id= '6847';
select COUNT(*)
from dv.notes_message
where user_id='13169'
  and agency_id_id= '13169'
  and notice_id= '13169'
  and route_id= '13169';
select agency_id
from m.agency
where agency_id_id= '16022'
  and valid_now=13746;
select COUNT(*)
from dv.notes_message
where user_id='14353'
  and agency_id_id= '14353'
  and notice_id= '14353'
  and route_id= '14353';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14568'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12372'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3945'
  and valid_now=9995;
select user_id
from m.agency
where valid_now=7023
  and agency_id_id= '19629';
select COUNT(*)
from dv.notes_message
where user_id='5860'
  and agency_id_id= '5860'
  and notice_id= '5860'
  and route_id= '5860';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14910'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5057
  and agency_id_id= '12527';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3246'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7452'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15811'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1052'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17588'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1806'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12925'
  and valid_now=2924;
select agency_id
from m.agency
where agency_id_id= '624'
  and valid_now=17415;
select user_id
from m.agency
where valid_now=10566
  and agency_id_id= '16768';
select user_id
from m.agency
where valid_now=16185
  and agency_id_id= '16649';
select COUNT(*)
from dv.notes_message
where user_id='7878'
  and agency_id_id= '7878'
  and notice_id= '7878'
  and route_id= '7878';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3211'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1707'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16364'
  and agency_id_id= '16364'
  and notice_id= '16364'
  and route_id= '16364';
select agency_id
from m.agency
where agency_id_id= '14824'
  and valid_now=4828;
select agency_id
from m.agency
where agency_id_id= '14821'
  and valid_now=1732;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3578'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3258'
  and valid_now=1408;
select COUNT(*)
from dv.notes_message
where user_id='2869'
  and agency_id_id= '2869'
  and notice_id= '2869'
  and route_id= '2869';
select COUNT(*)
from dv.notes_message
where user_id='7183'
  and agency_id_id= '7183'
  and notice_id= '7183'
  and route_id= '7183';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16093'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7177'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2598'
  and valid_now=10260;
select user_id
from m.agency
where valid_now=859
  and agency_id_id= '6212';
select COUNT(*)
from dv.notes_message
where user_id='301'
  and agency_id_id= '301'
  and notice_id= '301'
  and route_id= '301';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13757'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1597'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4931'
  and valid_now=10682;
select agency_id
from m.agency
where agency_id_id= '10773'
  and valid_now=15314;
select user_id
from m.agency
where valid_now=15685
  and agency_id_id= '11642';
select COUNT(*)
from dv.notes_message
where user_id='4849'
  and agency_id_id= '4849'
  and notice_id= '4849'
  and route_id= '4849';
select COUNT(*)
from dv.notes_message
where user_id='1592'
  and agency_id_id= '1592'
  and notice_id= '1592'
  and route_id= '1592';
select user_id
from m.agency
where valid_now=8887
  and agency_id_id= '3445';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2283'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18582'
  and agency_id_id= '18582'
  and notice_id= '18582'
  and route_id= '18582';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10308'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4911'
  and valid_now=6142;
select user_id
from m.agency
where valid_now=18470
  and agency_id_id= '1055';
select COUNT(*)
from dv.notes_message
where user_id='9504'
  and agency_id_id= '9504'
  and notice_id= '9504'
  and route_id= '9504';
select COUNT(*)
from dv.notes_message
where user_id='9233'
  and agency_id_id= '9233'
  and notice_id= '9233'
  and route_id= '9233';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9340'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10485'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4037'
  and agency_id_id= '4037'
  and notice_id= '4037'
  and route_id= '4037';
select user_id
from m.agency
where valid_now=196
  and agency_id_id= '8560';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11025'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10648
  and agency_id_id= '6828';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10912'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7347'
  and valid_now=12263;
select user_id
from m.agency
where valid_now=1880
  and agency_id_id= '1573';
select COUNT(*)
from dv.notes_message
where user_id='3303'
  and agency_id_id= '3303'
  and notice_id= '3303'
  and route_id= '3303';
select COUNT(*)
from dv.notes_message
where user_id='10675'
  and agency_id_id= '10675'
  and notice_id= '10675'
  and route_id= '10675';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9899'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1010
  and agency_id_id= '9639';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16939'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11552'
  and valid_now=5229;
select COUNT(*)
from dv.notes_message
where user_id='8063'
  and agency_id_id= '8063'
  and notice_id= '8063'
  and route_id= '8063';
select agency_id
from m.agency
where agency_id_id= '14957'
  and valid_now=15391;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6306'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3075'
  and valid_now=1399;
select agency_id
from m.agency
where agency_id_id= '7595'
  and valid_now=17257;
select agency_id
from m.agency
where agency_id_id= '7735'
  and valid_now=2082;
select COUNT(*)
from dv.notes_message
where user_id='1466'
  and agency_id_id= '1466'
  and notice_id= '1466'
  and route_id= '1466';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9160'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15693'
  and valid_now=17542;
select COUNT(*)
from dv.notes_message
where user_id='2188'
  and agency_id_id= '2188'
  and notice_id= '2188'
  and route_id= '2188';
select agency_id
from m.agency
where agency_id_id= '1043'
  and valid_now=6172;
select COUNT(*)
from dv.notes_message
where user_id='5469'
  and agency_id_id= '5469'
  and notice_id= '5469'
  and route_id= '5469';
select COUNT(*)
from dv.notes_message
where user_id='12076'
  and agency_id_id= '12076'
  and notice_id= '12076'
  and route_id= '12076';
select COUNT(*)
from dv.notes_message
where user_id='16264'
  and agency_id_id= '16264'
  and notice_id= '16264'
  and route_id= '16264';
select agency_id
from m.agency
where agency_id_id= '556'
  and valid_now=2526;
select agency_id
from m.agency
where agency_id_id= '18978'
  and valid_now=10034;
select COUNT(*)
from dv.notes_message
where user_id='1569'
  and agency_id_id= '1569'
  and notice_id= '1569'
  and route_id= '1569';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18933'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16076'
  and agency_id_id= '16076'
  and notice_id= '16076'
  and route_id= '16076';
select COUNT(*)
from dv.notes_message
where user_id='4345'
  and agency_id_id= '4345'
  and notice_id= '4345'
  and route_id= '4345';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19163'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19636'
  and agency_id_id= '19636'
  and notice_id= '19636'
  and route_id= '19636';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '33'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1258
  and agency_id_id= '19352';
select COUNT(*)
from dv.notes_message
where user_id='2087'
  and agency_id_id= '2087'
  and notice_id= '2087'
  and route_id= '2087';
select user_id
from m.agency
where valid_now=12077
  and agency_id_id= '11626';
select agency_id
from m.agency
where agency_id_id= '3639'
  and valid_now=7926;
select agency_id
from m.agency
where agency_id_id= '17401'
  and valid_now=12819;
select agency_id
from m.agency
where agency_id_id= '7186'
  and valid_now=14310;
select COUNT(*)
from dv.notes_message
where user_id='12313'
  and agency_id_id= '12313'
  and notice_id= '12313'
  and route_id= '12313';
select COUNT(*)
from dv.notes_message
where user_id='6084'
  and agency_id_id= '6084'
  and notice_id= '6084'
  and route_id= '6084';
select agency_id
from m.agency
where agency_id_id= '2206'
  and valid_now=13396;
select user_id
from m.agency
where valid_now=19729
  and agency_id_id= '13115';
select COUNT(*)
from dv.notes_message
where user_id='139'
  and agency_id_id= '139'
  and notice_id= '139'
  and route_id= '139';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13823'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16428
  and agency_id_id= '6434';
select COUNT(*)
from dv.notes_message
where user_id='7244'
  and agency_id_id= '7244'
  and notice_id= '7244'
  and route_id= '7244';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15673'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11509
  and agency_id_id= '5475';
select COUNT(*)
from dv.notes_message
where user_id='12075'
  and agency_id_id= '12075'
  and notice_id= '12075'
  and route_id= '12075';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8015'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7212'
  and agency_id_id= '7212'
  and notice_id= '7212'
  and route_id= '7212';
select agency_id
from m.agency
where agency_id_id= '18881'
  and valid_now=10930;
select COUNT(*)
from dv.notes_message
where user_id='10137'
  and agency_id_id= '10137'
  and notice_id= '10137'
  and route_id= '10137';
select COUNT(*)
from dv.notes_message
where user_id='4413'
  and agency_id_id= '4413'
  and notice_id= '4413'
  and route_id= '4413';
select COUNT(*)
from dv.notes_message
where user_id='16088'
  and agency_id_id= '16088'
  and notice_id= '16088'
  and route_id= '16088';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19891'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12040'
  and valid_now=12381;
select user_id
from m.agency
where valid_now=12515
  and agency_id_id= '17768';
select user_id
from m.agency
where valid_now=4773
  and agency_id_id= '13531';
select COUNT(*)
from dv.notes_message
where user_id='7256'
  and agency_id_id= '7256'
  and notice_id= '7256'
  and route_id= '7256';
select COUNT(*)
from dv.notes_message
where user_id='18622'
  and agency_id_id= '18622'
  and notice_id= '18622'
  and route_id= '18622';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11822'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=233
  and agency_id_id= '1306';
select user_id
from m.agency
where valid_now=4018
  and agency_id_id= '12431';
select COUNT(*)
from dv.notes_message
where user_id='4213'
  and agency_id_id= '4213'
  and notice_id= '4213'
  and route_id= '4213';
select agency_id
from m.agency
where agency_id_id= '3947'
  and valid_now=14241;
select agency_id
from m.agency
where agency_id_id= '8200'
  and valid_now=18100;
select COUNT(*)
from dv.notes_message
where user_id='10456'
  and agency_id_id= '10456'
  and notice_id= '10456'
  and route_id= '10456';
select user_id
from m.agency
where valid_now=15126
  and agency_id_id= '12970';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9249'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11886'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16880'
  and valid_now=11230;
select agency_id
from m.agency
where agency_id_id= '14121'
  and valid_now=1105;
select COUNT(*)
from dv.notes_message
where user_id='17234'
  and agency_id_id= '17234'
  and notice_id= '17234'
  and route_id= '17234';
select COUNT(*)
from dv.notes_message
where user_id='11696'
  and agency_id_id= '11696'
  and notice_id= '11696'
  and route_id= '11696';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2463'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16094'
  and valid_now=14140;
select COUNT(*)
from dv.notes_message
where user_id='19551'
  and agency_id_id= '19551'
  and notice_id= '19551'
  and route_id= '19551';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13534'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13941'
  and valid_now=11945;
select COUNT(*)
from dv.notes_message
where user_id='5891'
  and agency_id_id= '5891'
  and notice_id= '5891'
  and route_id= '5891';
select agency_id
from m.agency
where agency_id_id= '13438'
  and valid_now=1600;
select user_id
from m.agency
where valid_now=12896
  and agency_id_id= '36';
select COUNT(*)
from dv.notes_message
where user_id='8152'
  and agency_id_id= '8152'
  and notice_id= '8152'
  and route_id= '8152';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16675'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17537'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19233'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17025'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9353
  and agency_id_id= '6937';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13940'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3924'
  and agency_id_id= '3924'
  and notice_id= '3924'
  and route_id= '3924';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8682'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19287'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10939'
  and valid_now=384;
select user_id
from m.agency
where valid_now=19833
  and agency_id_id= '5875';
select user_id
from m.agency
where valid_now=15652
  and agency_id_id= '5584';
select agency_id
from m.agency
where agency_id_id= '1111'
  and valid_now=8660;
select agency_id
from m.agency
where agency_id_id= '16131'
  and valid_now=11524;
select user_id
from m.agency
where valid_now=6072
  and agency_id_id= '11681';
select agency_id
from m.agency
where agency_id_id= '9030'
  and valid_now=12025;
select agency_id
from m.agency
where agency_id_id= '4422'
  and valid_now=12706;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3646'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1666'
  and valid_now=16211;
select agency_id
from m.agency
where agency_id_id= '12887'
  and valid_now=4806;
select user_id
from m.agency
where valid_now=5379
  and agency_id_id= '6489';
select user_id
from m.agency
where valid_now=7188
  and agency_id_id= '12038';
select user_id
from m.agency
where valid_now=19037
  and agency_id_id= '2838';
select COUNT(*)
from dv.notes_message
where user_id='17302'
  and agency_id_id= '17302'
  and notice_id= '17302'
  and route_id= '17302';
select agency_id
from m.agency
where agency_id_id= '11969'
  and valid_now=7858;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4011'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15381'
  and agency_id_id= '15381'
  and notice_id= '15381'
  and route_id= '15381';
select COUNT(*)
from dv.notes_message
where user_id='11624'
  and agency_id_id= '11624'
  and notice_id= '11624'
  and route_id= '11624';
select user_id
from m.agency
where valid_now=4735
  and agency_id_id= '523';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14522'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13465'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2822
  and agency_id_id= '15179';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18413'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=595
  and agency_id_id= '19880';
select COUNT(*)
from dv.notes_message
where user_id='8082'
  and agency_id_id= '8082'
  and notice_id= '8082'
  and route_id= '8082';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17267'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5494'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4700'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7235'
  and valid_now=488;
select agency_id
from m.agency
where agency_id_id= '6003'
  and valid_now=3251;
select user_id
from m.agency
where valid_now=5781
  and agency_id_id= '199';
select agency_id
from m.agency
where agency_id_id= '14117'
  and valid_now=1180;
select agency_id
from m.agency
where agency_id_id= '1794'
  and valid_now=11945;
select agency_id
from m.agency
where agency_id_id= '2986'
  and valid_now=3954;
select agency_id
from m.agency
where agency_id_id= '12630'
  and valid_now=9068;
select COUNT(*)
from dv.notes_message
where user_id='19262'
  and agency_id_id= '19262'
  and notice_id= '19262'
  and route_id= '19262';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2838'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9091'
  and valid_now=18259;
select user_id
from m.agency
where valid_now=10151
  and agency_id_id= '9094';
select COUNT(*)
from dv.notes_message
where user_id='14077'
  and agency_id_id= '14077'
  and notice_id= '14077'
  and route_id= '14077';
select COUNT(*)
from dv.notes_message
where user_id='2139'
  and agency_id_id= '2139'
  and notice_id= '2139'
  and route_id= '2139';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '948'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8011'
  and agency_id_id= '8011'
  and notice_id= '8011'
  and route_id= '8011';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18767'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '837'
  and valid_now=4180;
select agency_id
from m.agency
where agency_id_id= '3751'
  and valid_now=11927;
select user_id
from m.agency
where valid_now=3899
  and agency_id_id= '13038';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12736'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18001'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8385
  and agency_id_id= '4408';
select user_id
from m.agency
where valid_now=18105
  and agency_id_id= '11785';
select user_id
from m.agency
where valid_now=1470
  and agency_id_id= '9140';
select user_id
from m.agency
where valid_now=18104
  and agency_id_id= '7654';
select user_id
from m.agency
where valid_now=9953
  and agency_id_id= '17648';
select COUNT(*)
from dv.notes_message
where user_id='14990'
  and agency_id_id= '14990'
  and notice_id= '14990'
  and route_id= '14990';
select COUNT(*)
from dv.notes_message
where user_id='1366'
  and agency_id_id= '1366'
  and notice_id= '1366'
  and route_id= '1366';
select COUNT(*)
from dv.notes_message
where user_id='10579'
  and agency_id_id= '10579'
  and notice_id= '10579'
  and route_id= '10579';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4181'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10165'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18448'
  and valid_now=13032;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12491'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14466'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16038'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19487'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8649'
  and valid_now=18100;
select user_id
from m.agency
where valid_now=5147
  and agency_id_id= '8571';
select COUNT(*)
from dv.notes_message
where user_id='14575'
  and agency_id_id= '14575'
  and notice_id= '14575'
  and route_id= '14575';
select COUNT(*)
from dv.notes_message
where user_id='14884'
  and agency_id_id= '14884'
  and notice_id= '14884'
  and route_id= '14884';
select COUNT(*)
from dv.notes_message
where user_id='6902'
  and agency_id_id= '6902'
  and notice_id= '6902'
  and route_id= '6902';
select user_id
from m.agency
where valid_now=1855
  and agency_id_id= '19604';
select COUNT(*)
from dv.notes_message
where user_id='12890'
  and agency_id_id= '12890'
  and notice_id= '12890'
  and route_id= '12890';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10247'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19648'
  and valid_now=8507;
select agency_id
from m.agency
where agency_id_id= '6489'
  and valid_now=16527;
select agency_id
from m.agency
where agency_id_id= '15828'
  and valid_now=9453;
select user_id
from m.agency
where valid_now=7812
  and agency_id_id= '1152';
select user_id
from m.agency
where valid_now=8270
  and agency_id_id= '13980';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '148'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16929'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '184'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9680'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15497'
  and agency_id_id= '15497'
  and notice_id= '15497'
  and route_id= '15497';
select COUNT(*)
from dv.notes_message
where user_id='7500'
  and agency_id_id= '7500'
  and notice_id= '7500'
  and route_id= '7500';
select user_id
from m.agency
where valid_now=7685
  and agency_id_id= '136';
select user_id
from m.agency
where valid_now=8712
  and agency_id_id= '6537';
select COUNT(*)
from dv.notes_message
where user_id='19222'
  and agency_id_id= '19222'
  and notice_id= '19222'
  and route_id= '19222';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15174'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4608'
  and valid_now=17070;
select agency_id
from m.agency
where agency_id_id= '11821'
  and valid_now=4420;
select agency_id
from m.agency
where agency_id_id= '13911'
  and valid_now=11425;
select user_id
from m.agency
where valid_now=10001
  and agency_id_id= '2083';
select agency_id
from m.agency
where agency_id_id= '11057'
  and valid_now=1857;
select COUNT(*)
from dv.notes_message
where user_id='18966'
  and agency_id_id= '18966'
  and notice_id= '18966'
  and route_id= '18966';
select user_id
from m.agency
where valid_now=1462
  and agency_id_id= '58';
select user_id
from m.agency
where valid_now=4240
  and agency_id_id= '17421';
select COUNT(*)
from dv.notes_message
where user_id='5391'
  and agency_id_id= '5391'
  and notice_id= '5391'
  and route_id= '5391';
select COUNT(*)
from dv.notes_message
where user_id='1007'
  and agency_id_id= '1007'
  and notice_id= '1007'
  and route_id= '1007';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17974'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9175
  and agency_id_id= '12360';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1757'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1465'
  and valid_now=5973;
select user_id
from m.agency
where valid_now=17682
  and agency_id_id= '19347';
select COUNT(*)
from dv.notes_message
where user_id='12134'
  and agency_id_id= '12134'
  and notice_id= '12134'
  and route_id= '12134';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16247'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11541'
  and valid_now=1206;
select agency_id
from m.agency
where agency_id_id= '10458'
  and valid_now=3658;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11583'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7129
  and agency_id_id= '10094';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4314'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='765'
  and agency_id_id= '765'
  and notice_id= '765'
  and route_id= '765';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3205'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2302'
  and valid_now=10396;
select user_id
from m.agency
where valid_now=13749
  and agency_id_id= '12043';
select user_id
from m.agency
where valid_now=2286
  and agency_id_id= '4348';
select COUNT(*)
from dv.notes_message
where user_id='14644'
  and agency_id_id= '14644'
  and notice_id= '14644'
  and route_id= '14644';
select agency_id
from m.agency
where agency_id_id= '13759'
  and valid_now=16132;
select user_id
from m.agency
where valid_now=16948
  and agency_id_id= '2510';
select COUNT(*)
from dv.notes_message
where user_id='1189'
  and agency_id_id= '1189'
  and notice_id= '1189'
  and route_id= '1189';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2455'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4833'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=122
  and agency_id_id= '5253';
select COUNT(*)
from dv.notes_message
where user_id='5375'
  and agency_id_id= '5375'
  and notice_id= '5375'
  and route_id= '5375';
select COUNT(*)
from dv.notes_message
where user_id='19756'
  and agency_id_id= '19756'
  and notice_id= '19756'
  and route_id= '19756';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12370'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18054';
select user_id
from m.agency
where valid_now=10784
  and agency_id_id= '19800';
select COUNT(*)
from dv.notes_message
where user_id='11786'
  and agency_id_id= '11786'
  and notice_id= '11786'
  and route_id= '11786';
select COUNT(*)
from dv.notes_message
where user_id='15831'
  and agency_id_id= '15831'
  and notice_id= '15831'
  and route_id= '15831';
select agency_id
from m.agency
where agency_id_id= '7559'
  and valid_now=6787;
select agency_id
from m.agency
where agency_id_id= '15771'
  and valid_now=12418;
select agency_id
from m.agency
where agency_id_id= '3771'
  and valid_now=3563;
select agency_id
from m.agency
where agency_id_id= '6022'
  and valid_now=12722;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10853'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11728'
  and valid_now=13031;
select agency_id
from m.agency
where agency_id_id= '3331'
  and valid_now=3323;
select agency_id
from m.agency
where agency_id_id= '1031'
  and valid_now=9198;
select agency_id
from m.agency
where agency_id_id= '19435'
  and valid_now=19522;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4377'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17122'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8932'
  and valid_now=1933;
select agency_id
from m.agency
where agency_id_id= '18921'
  and valid_now=3058;
select COUNT(*)
from dv.notes_message
where user_id='8468'
  and agency_id_id= '8468'
  and notice_id= '8468'
  and route_id= '8468';
select COUNT(*)
from dv.notes_message
where user_id='7195'
  and agency_id_id= '7195'
  and notice_id= '7195'
  and route_id= '7195';
select COUNT(*)
from dv.notes_message
where user_id='8038'
  and agency_id_id= '8038'
  and notice_id= '8038'
  and route_id= '8038';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16744'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8076
  and agency_id_id= '9412';
select COUNT(*)
from dv.notes_message
where user_id='16091'
  and agency_id_id= '16091'
  and notice_id= '16091'
  and route_id= '16091';
select agency_id
from m.agency
where agency_id_id= '13832'
  and valid_now=14508;
select user_id
from m.agency
where valid_now=359
  and agency_id_id= '16814';
select user_id
from m.agency
where valid_now=7843
  and agency_id_id= '2473';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17133'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11185'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15023
  and agency_id_id= '2768';
select COUNT(*)
from dv.notes_message
where user_id='2999'
  and agency_id_id= '2999'
  and notice_id= '2999'
  and route_id= '2999';
select user_id
from m.agency
where valid_now=12608
  and agency_id_id= '19330';
select user_id
from m.agency
where valid_now=11780
  and agency_id_id= '10141';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6269'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12796
  and agency_id_id= '2090';
select COUNT(*)
from dv.notes_message
where user_id='19162'
  and agency_id_id= '19162'
  and notice_id= '19162'
  and route_id= '19162';
select COUNT(*)
from dv.notes_message
where user_id='428'
  and agency_id_id= '428'
  and notice_id= '428'
  and route_id= '428';
select agency_id
from m.agency
where agency_id_id= '8146'
  and valid_now=18412;
select agency_id
from m.agency
where agency_id_id= '15119'
  and valid_now=5067;
select COUNT(*)
from dv.notes_message
where user_id='7334'
  and agency_id_id= '7334'
  and notice_id= '7334'
  and route_id= '7334';
select COUNT(*)
from dv.notes_message
where user_id='12321'
  and agency_id_id= '12321'
  and notice_id= '12321'
  and route_id= '12321';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10860'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19179'
  and valid_now=13383;
select agency_id
from m.agency
where agency_id_id= '15196'
  and valid_now=2180;
select user_id
from m.agency
where valid_now=340
  and agency_id_id= '15403';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2976'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10134'
  and valid_now=15108;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1040'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7927
  and agency_id_id= '7292';
select COUNT(*)
from dv.notes_message
where user_id='19465'
  and agency_id_id= '19465'
  and notice_id= '19465'
  and route_id= '19465';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11705'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13119
  and agency_id_id= '16498';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4617'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4266
  and agency_id_id= '5712';
select COUNT(*)
from dv.notes_message
where user_id='6179'
  and agency_id_id= '6179'
  and notice_id= '6179'
  and route_id= '6179';
select COUNT(*)
from dv.notes_message
where user_id='16423'
  and agency_id_id= '16423'
  and notice_id= '16423'
  and route_id= '16423';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4731'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3045
  and agency_id_id= '8309';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12593'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14233'
  and valid_now=3474;
select agency_id
from m.agency
where agency_id_id= '11446'
  and valid_now=17183;
select user_id
from m.agency
where valid_now=7715
  and agency_id_id= '9319';
select user_id
from m.agency
where valid_now=15894
  and agency_id_id= '14571';
select COUNT(*)
from dv.notes_message
where user_id='16780'
  and agency_id_id= '16780'
  and notice_id= '16780'
  and route_id= '16780';
select a.agency_timezone
from m.agency a
where a.agency_id = '17031';
select a.agency_timezone
from m.agency a
where a.agency_id = '15523';
select COUNT(*)
from dv.notes_message
where user_id='12089'
  and agency_id_id= '12089'
  and notice_id= '12089'
  and route_id= '12089';
select COUNT(*)
from dv.notes_message
where user_id='5248'
  and agency_id_id= '5248'
  and notice_id= '5248'
  and route_id= '5248';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12389'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11682'
  and valid_now=1368;
select user_id
from m.agency
where valid_now=3482
  and agency_id_id= '13168';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12957'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19726'
  and valid_now=12726;
select user_id
from m.agency
where valid_now=2056
  and agency_id_id= '14551';
select user_id
from m.agency
where valid_now=14417
  and agency_id_id= '3538';
select a.agency_timezone
from m.agency a
where a.agency_id = '19000';
select a.agency_timezone
from m.agency a
where a.agency_id = '8504';
select user_id
from m.agency
where valid_now=18665
  and agency_id_id= '18317';
select a.agency_timezone
from m.agency a
where a.agency_id = '9613';
select a.agency_timezone
from m.agency a
where a.agency_id = '16009';
select agency_id
from m.agency
where agency_id_id= '6739'
  and valid_now=1190;
select COUNT(*)
from dv.notes_message
where user_id='1062'
  and agency_id_id= '1062'
  and notice_id= '1062'
  and route_id= '1062';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18157'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17354'
  and valid_now=11269;
select agency_id
from m.agency
where agency_id_id= '8726'
  and valid_now=8777;
select agency_id
from m.agency
where agency_id_id= '2892'
  and valid_now=17212;
select COUNT(*)
from dv.notes_message
where user_id='8648'
  and agency_id_id= '8648'
  and notice_id= '8648'
  and route_id= '8648';
select user_id
from m.agency
where valid_now=6859
  and agency_id_id= '421';
select COUNT(*)
from dv.notes_message
where user_id='19962'
  and agency_id_id= '19962'
  and notice_id= '19962'
  and route_id= '19962';
select a.agency_timezone
from m.agency a
where a.agency_id = '1591';
select agency_id
from m.agency
where agency_id_id= '11365'
  and valid_now=14868;
select COUNT(*)
from dv.notes_message
where user_id='16452'
  and agency_id_id= '16452'
  and notice_id= '16452'
  and route_id= '16452';
select COUNT(*)
from dv.notes_message
where user_id='866'
  and agency_id_id= '866'
  and notice_id= '866'
  and route_id= '866';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6117'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7749'
  and agency_id_id= '7749'
  and notice_id= '7749'
  and route_id= '7749';
select COUNT(*)
from dv.notes_message
where user_id='552'
  and agency_id_id= '552'
  and notice_id= '552'
  and route_id= '552';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15226'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7473'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9201'
  and valid_now=4521;
select COUNT(*)
from dv.notes_message
where user_id='4713'
  and agency_id_id= '4713'
  and notice_id= '4713'
  and route_id= '4713';
select COUNT(*)
from dv.notes_message
where user_id='9037'
  and agency_id_id= '9037'
  and notice_id= '9037'
  and route_id= '9037';
select COUNT(*)
from dv.notes_message
where user_id='484'
  and agency_id_id= '484'
  and notice_id= '484'
  and route_id= '484';
select agency_id
from m.agency
where agency_id_id= '19994'
  and valid_now=17853;
select agency_id
from m.agency
where agency_id_id= '6895'
  and valid_now=19713;
select user_id
from m.agency
where valid_now=8155
  and agency_id_id= '4929';
select agency_id
from m.agency
where agency_id_id= '11625'
  and valid_now=7628;
select agency_id
from m.agency
where agency_id_id= '11100'
  and valid_now=10915;
select user_id
from m.agency
where valid_now=18090
  and agency_id_id= '15384';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3168'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3891
  and agency_id_id= '12383';
select user_id
from m.agency
where valid_now=9177
  and agency_id_id= '15678';
select COUNT(*)
from dv.notes_message
where user_id='4'
  and agency_id_id= '4'
  and notice_id= '4'
  and route_id= '4';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3869'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12701
  and agency_id_id= '9150';
select COUNT(*)
from dv.notes_message
where user_id='18947'
  and agency_id_id= '18947'
  and notice_id= '18947'
  and route_id= '18947';
select agency_id
from m.agency
where agency_id_id= '5383'
  and valid_now=6249;
select user_id
from m.agency
where valid_now=5118
  and agency_id_id= '16197';
select COUNT(*)
from dv.notes_message
where user_id='7741'
  and agency_id_id= '7741'
  and notice_id= '7741'
  and route_id= '7741';
select COUNT(*)
from dv.notes_message
where user_id='360'
  and agency_id_id= '360'
  and notice_id= '360'
  and route_id= '360';
select agency_id
from m.agency
where agency_id_id= '19853'
  and valid_now=14117;
select user_id
from m.agency
where valid_now=8937
  and agency_id_id= '18302';
select COUNT(*)
from dv.notes_message
where user_id='12592'
  and agency_id_id= '12592'
  and notice_id= '12592'
  and route_id= '12592';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6120'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3470'
  and agency_id_id= '3470'
  and notice_id= '3470'
  and route_id= '3470';
select agency_id
from m.agency
where agency_id_id= '3229'
  and valid_now=5549;
select agency_id
from m.agency
where agency_id_id= '5001'
  and valid_now=2515;
select COUNT(*)
from dv.notes_message
where user_id='3715'
  and agency_id_id= '3715'
  and notice_id= '3715'
  and route_id= '3715';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8300'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16076'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19429
  and agency_id_id= '14166';
select agency_id
from m.agency
where agency_id_id= '6744'
  and valid_now=457;
select user_id
from m.agency
where valid_now=3106
  and agency_id_id= '6495';
select COUNT(*)
from dv.notes_message
where user_id='18101'
  and agency_id_id= '18101'
  and notice_id= '18101'
  and route_id= '18101';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5392'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6665'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19281'
  and valid_now=13364;
select user_id
from m.agency
where valid_now=3105
  and agency_id_id= '174';
select COUNT(*)
from dv.notes_message
where user_id='4700'
  and agency_id_id= '4700'
  and notice_id= '4700'
  and route_id= '4700';
select agency_id
from m.agency
where agency_id_id= '14246'
  and valid_now=16926;
select agency_id
from m.agency
where agency_id_id= '7228'
  and valid_now=9460;
select COUNT(*)
from dv.notes_message
where user_id='15416'
  and agency_id_id= '15416'
  and notice_id= '15416'
  and route_id= '15416';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1095'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14997'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19373'
  and agency_id_id= '19373'
  and notice_id= '19373'
  and route_id= '19373';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2050'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12817
  and agency_id_id= '11274';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3705'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2313'
  and valid_now=14067;
select user_id
from m.agency
where valid_now=5649
  and agency_id_id= '15220';
select COUNT(*)
from dv.notes_message
where user_id='191'
  and agency_id_id= '191'
  and notice_id= '191'
  and route_id= '191';
select COUNT(*)
from dv.notes_message
where user_id='14536'
  and agency_id_id= '14536'
  and notice_id= '14536'
  and route_id= '14536';
select agency_id
from m.agency
where agency_id_id= '17590'
  and valid_now=10078;
select a.agency_timezone
from m.agency a
where a.agency_id = '18377';
select agency_id
from m.agency
where agency_id_id= '19977'
  and valid_now=1539;
select COUNT(*)
from dv.notes_message
where user_id='4775'
  and agency_id_id= '4775'
  and notice_id= '4775'
  and route_id= '4775';
select COUNT(*)
from dv.notes_message
where user_id='4340'
  and agency_id_id= '4340'
  and notice_id= '4340'
  and route_id= '4340';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '621'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1828'
  and valid_now=7358;
select user_id
from m.agency
where valid_now=1760
  and agency_id_id= '12095';
select COUNT(*)
from dv.notes_message
where user_id='15704'
  and agency_id_id= '15704'
  and notice_id= '15704'
  and route_id= '15704';
select COUNT(*)
from dv.notes_message
where user_id='11479'
  and agency_id_id= '11479'
  and notice_id= '11479'
  and route_id= '11479';
select agency_id
from m.agency
where agency_id_id= '7173'
  and valid_now=4412;
select user_id
from m.agency
where valid_now=14874
  and agency_id_id= '19677';
select user_id
from m.agency
where valid_now=11038
  and agency_id_id= '1662';
select COUNT(*)
from dv.notes_message
where user_id='19210'
  and agency_id_id= '19210'
  and notice_id= '19210'
  and route_id= '19210';
select COUNT(*)
from dv.notes_message
where user_id='4704'
  and agency_id_id= '4704'
  and notice_id= '4704'
  and route_id= '4704';
select COUNT(*)
from dv.notes_message
where user_id='10592'
  and agency_id_id= '10592'
  and notice_id= '10592'
  and route_id= '10592';
select user_id
from m.agency
where valid_now=19685
  and agency_id_id= '9777';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13319'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12227
  and agency_id_id= '15150';
select COUNT(*)
from dv.notes_message
where user_id='289'
  and agency_id_id= '289'
  and notice_id= '289'
  and route_id= '289';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19015'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9564'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16918'
  and agency_id_id= '16918'
  and notice_id= '16918'
  and route_id= '16918';
select COUNT(*)
from dv.notes_message
where user_id='16906'
  and agency_id_id= '16906'
  and notice_id= '16906'
  and route_id= '16906';
select COUNT(*)
from dv.notes_message
where user_id='19701'
  and agency_id_id= '19701'
  and notice_id= '19701'
  and route_id= '19701';
select agency_id
from m.agency
where agency_id_id= '288'
  and valid_now=13182;
select COUNT(*)
from dv.notes_message
where user_id='12696'
  and agency_id_id= '12696'
  and notice_id= '12696'
  and route_id= '12696';
select COUNT(*)
from dv.notes_message
where user_id='18538'
  and agency_id_id= '18538'
  and notice_id= '18538'
  and route_id= '18538';
select COUNT(*)
from dv.notes_message
where user_id='941'
  and agency_id_id= '941'
  and notice_id= '941'
  and route_id= '941';
select agency_id
from m.agency
where agency_id_id= '13426'
  and valid_now=15510;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10297'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19184'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14341'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3806
  and agency_id_id= '13187';
select user_id
from m.agency
where valid_now=6301
  and agency_id_id= '19756';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10327'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15871'
  and valid_now=15488;
select agency_id
from m.agency
where agency_id_id= '3643'
  and valid_now=8737;
select agency_id
from m.agency
where agency_id_id= '1012'
  and valid_now=17769;
select COUNT(*)
from dv.notes_message
where user_id='9489'
  and agency_id_id= '9489'
  and notice_id= '9489'
  and route_id= '9489';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8998'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6768'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5458
  and agency_id_id= '1401';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7876'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8429'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3350
  and agency_id_id= '3202';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8730'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18212
  and agency_id_id= '4685';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10131'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2994'
  and valid_now=4300;
select user_id
from m.agency
where valid_now=14253
  and agency_id_id= '3583';
select user_id
from m.agency
where valid_now=4765
  and agency_id_id= '12710';
select agency_id
from m.agency
where agency_id_id= '15094'
  and valid_now=9573;
select user_id
from m.agency
where valid_now=17484
  and agency_id_id= '176';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10026'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6065'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1191'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1013'
  and valid_now=6679;
select user_id
from m.agency
where valid_now=16864
  and agency_id_id= '15879';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8540'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3897'
  and valid_now=3656;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8559'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15573'
  and valid_now=7653;
select user_id
from m.agency
where valid_now=427
  and agency_id_id= '594';
select COUNT(*)
from dv.notes_message
where user_id='4356'
  and agency_id_id= '4356'
  and notice_id= '4356'
  and route_id= '4356';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17129'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9289'
  and valid_now=10561;
select user_id
from m.agency
where valid_now=1119
  and agency_id_id= '708';
select user_id
from m.agency
where valid_now=10028
  and agency_id_id= '8154';
select agency_id
from m.agency
where agency_id_id= '1451'
  and valid_now=14985;
select user_id
from m.agency
where valid_now=5989
  and agency_id_id= '11462';
select user_id
from m.agency
where valid_now=2052
  and agency_id_id= '3708';
select user_id
from m.agency
where valid_now=2572
  and agency_id_id= '19551';
select COUNT(*)
from dv.notes_message
where user_id='8253'
  and agency_id_id= '8253'
  and notice_id= '8253'
  and route_id= '8253';
select COUNT(*)
from dv.notes_message
where user_id='17118'
  and agency_id_id= '17118'
  and notice_id= '17118'
  and route_id= '17118';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12750'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9320'
  and valid_now=15328;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13731'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6083'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9057'
  and valid_now=6150;
select agency_id
from m.agency
where agency_id_id= '879'
  and valid_now=6490;
select user_id
from m.agency
where valid_now=16599
  and agency_id_id= '15623';
select user_id
from m.agency
where valid_now=330
  and agency_id_id= '12765';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4179'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16782'
  and valid_now=18987;
select user_id
from m.agency
where valid_now=18230
  and agency_id_id= '10755';
select user_id
from m.agency
where valid_now=18930
  and agency_id_id= '14550';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17251'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13379
  and agency_id_id= '16393';
select COUNT(*)
from dv.notes_message
where user_id='16429'
  and agency_id_id= '16429'
  and notice_id= '16429'
  and route_id= '16429';
select COUNT(*)
from dv.notes_message
where user_id='13380'
  and agency_id_id= '13380'
  and notice_id= '13380'
  and route_id= '13380';
select COUNT(*)
from dv.notes_message
where user_id='17879'
  and agency_id_id= '17879'
  and notice_id= '17879'
  and route_id= '17879';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13629'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16896'
  and valid_now=9973;
select agency_id
from m.agency
where agency_id_id= '5044'
  and valid_now=6139;
select agency_id
from m.agency
where agency_id_id= '1541'
  and valid_now=17312;
select agency_id
from m.agency
where agency_id_id= '9108'
  and valid_now=19764;
select user_id
from m.agency
where valid_now=6465
  and agency_id_id= '12903';
select COUNT(*)
from dv.notes_message
where user_id='15031'
  and agency_id_id= '15031'
  and notice_id= '15031'
  and route_id= '15031';
select agency_id
from m.agency
where agency_id_id= '12985'
  and valid_now=16823;
select agency_id
from m.agency
where agency_id_id= '8999'
  and valid_now=11253;
select user_id
from m.agency
where valid_now=16072
  and agency_id_id= '6935';
select COUNT(*)
from dv.notes_message
where user_id='9716'
  and agency_id_id= '9716'
  and notice_id= '9716'
  and route_id= '9716';
select user_id
from m.agency
where valid_now=14602
  and agency_id_id= '14580';
select COUNT(*)
from dv.notes_message
where user_id='3256'
  and agency_id_id= '3256'
  and notice_id= '3256'
  and route_id= '3256';
select COUNT(*)
from dv.notes_message
where user_id='10103'
  and agency_id_id= '10103'
  and notice_id= '10103'
  and route_id= '10103';
select COUNT(*)
from dv.notes_message
where user_id='18943'
  and agency_id_id= '18943'
  and notice_id= '18943'
  and route_id= '18943';
select user_id
from m.agency
where valid_now=13638
  and agency_id_id= '12409';
select user_id
from m.agency
where valid_now=19706
  and agency_id_id= '16935';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_9978'
  and t.trip_id = 2540
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_18295'
  and t.trip_id = 1617
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19511
  and agency_id_id= '13568';
select COUNT(*)
from dv.notes_message
where user_id='17887'
  and agency_id_id= '17887'
  and notice_id= '17887'
  and route_id= '17887';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3538'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4455'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18373'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2324'
  and valid_now=13078;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10716'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_13621'
  and t.trip_id = 5640
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_16749'
  and t.trip_id = 14653
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4505'
  and valid_now=13953;
select COUNT(*)
from dv.notes_message
where user_id='18912'
  and agency_id_id= '18912'
  and notice_id= '18912'
  and route_id= '18912';
select agency_id
from m.agency
where agency_id_id= '4766'
  and valid_now=9525;
select COUNT(*)
from dv.notes_message
where user_id='2122'
  and agency_id_id= '2122'
  and notice_id= '2122'
  and route_id= '2122';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19190'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9864'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16626'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_15603'
  and t.trip_id = 7977
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7057'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '369'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_2686'
  and t.trip_id = 1516
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6870'
  and agency_id_id= '6870'
  and notice_id= '6870'
  and route_id= '6870';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_18368'
  and t.trip_id = 10190
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '95'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12892'
  and agency_id_id= '12892'
  and notice_id= '12892'
  and route_id= '12892';
select agency_id
from m.agency
where agency_id_id= '18209'
  and valid_now=19990;
select agency_id
from m.agency
where agency_id_id= '19682'
  and valid_now=12892;
select user_id
from m.agency
where valid_now=521
  and agency_id_id= '16220';
select COUNT(*)
from dv.notes_message
where user_id='7454'
  and agency_id_id= '7454'
  and notice_id= '7454'
  and route_id= '7454';
select agency_id
from m.agency
where agency_id_id= '6216'
  and valid_now=9136;
select agency_id
from m.agency
where agency_id_id= '8231'
  and valid_now=8172;
select user_id
from m.agency
where valid_now=7149
  and agency_id_id= '5144';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4239'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13066'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3809'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17518'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10323
  and agency_id_id= '10882';
select agency_id
from m.agency
where agency_id_id= '1302'
  and valid_now=11388;
select agency_id
from m.agency
where agency_id_id= '1829'
  and valid_now=14623;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9905'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9725'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3017'
  and valid_now=5624;
select COUNT(*)
from dv.notes_message
where user_id='7949'
  and agency_id_id= '7949'
  and notice_id= '7949'
  and route_id= '7949';
select agency_id
from m.agency
where agency_id_id= '9587'
  and valid_now=12102;
select COUNT(*)
from dv.notes_message
where user_id='19319'
  and agency_id_id= '19319'
  and notice_id= '19319'
  and route_id= '19319';
select COUNT(*)
from dv.notes_message
where user_id='19217'
  and agency_id_id= '19217'
  and notice_id= '19217'
  and route_id= '19217';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10151'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5818
  and agency_id_id= '6443';
select user_id
from m.agency
where valid_now=18411
  and agency_id_id= '9073';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_4795'
  and t.trip_id = 16700
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_570'
  and t.trip_id = 8126
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_1396'
  and t.trip_id = 11057
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_1022'
  and t.trip_id = 17041
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12877'
  and agency_id_id= '12877'
  and notice_id= '12877'
  and route_id= '12877';
select COUNT(*)
from dv.notes_message
where user_id='8142'
  and agency_id_id= '8142'
  and notice_id= '8142'
  and route_id= '8142';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_17656'
  and t.trip_id = 1999
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_13534'
  and t.trip_id = 2592
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13218'
  and valid_now=532;
select COUNT(*)
from dv.notes_message
where user_id='8923'
  and agency_id_id= '8923'
  and notice_id= '8923'
  and route_id= '8923';
select COUNT(*)
from dv.notes_message
where user_id='14766'
  and agency_id_id= '14766'
  and notice_id= '14766'
  and route_id= '14766';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18048'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13747
  and agency_id_id= '2039';
select a.agency_timezone
from m.agency a
where a.agency_id = '447';
select a.agency_timezone
from m.agency a
where a.agency_id = '15014';
select agency_id
from m.agency
where agency_id_id= '1629'
  and valid_now=3351;
select user_id
from m.agency
where valid_now=18804
  and agency_id_id= '2552';
select user_id
from m.agency
where valid_now=2221
  and agency_id_id= '3018';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9119'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=17760
  and agency_id_id= '10984';
select user_id
from m.agency
where valid_now=8359
  and agency_id_id= '8425';
select COUNT(*)
from dv.notes_message
where user_id='7185'
  and agency_id_id= '7185'
  and notice_id= '7185'
  and route_id= '7185';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17498'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3598'
  and valid_now=19383;
select COUNT(*)
from dv.notes_message
where user_id='8363'
  and agency_id_id= '8363'
  and notice_id= '8363'
  and route_id= '8363';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5035'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8902
  and agency_id_id= '12425';
select a.agency_timezone
from m.agency a
where a.agency_id = '1769';
select a.agency_timezone
from m.agency a
where a.agency_id = '1986';
select a.agency_timezone
from m.agency a
where a.agency_id = '977';
select a.agency_timezone
from m.agency a
where a.agency_id = '11498';
select agency_id
from m.agency
where agency_id_id= '12427'
  and valid_now=14417;
select agency_id
from m.agency
where agency_id_id= '15505'
  and valid_now=5404;
select agency_id
from m.agency
where agency_id_id= '8914'
  and valid_now=13964;
select COUNT(*)
from dv.notes_message
where user_id='1306'
  and agency_id_id= '1306'
  and notice_id= '1306'
  and route_id= '1306';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7187'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5844'
  and valid_now=3106;
select user_id
from m.agency
where valid_now=14401
  and agency_id_id= '908';
select user_id
from m.agency
where valid_now=15191
  and agency_id_id= '9409';
select user_id
from m.agency
where valid_now=3939
  and agency_id_id= '15511';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7454'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10824'
  and valid_now=14893;
select user_id
from m.agency
where valid_now=4198
  and agency_id_id= '10579';
select COUNT(*)
from dv.notes_message
where user_id='16556'
  and agency_id_id= '16556'
  and notice_id= '16556'
  and route_id= '16556';
select user_id
from m.agency
where valid_now=8462
  and agency_id_id= '14118';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15403'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9586'
  and valid_now=12354;
select user_id
from m.agency
where valid_now=16573
  and agency_id_id= '2040';
select user_id
from m.agency
where valid_now=3964
  and agency_id_id= '5423';
select COUNT(*)
from dv.notes_message
where user_id='19039'
  and agency_id_id= '19039'
  and notice_id= '19039'
  and route_id= '19039';
select COUNT(*)
from dv.notes_message
where user_id='3316'
  and agency_id_id= '3316'
  and notice_id= '3316'
  and route_id= '3316';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14510'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19113
  and agency_id_id= '15868';
select user_id
from m.agency
where valid_now=2542
  and agency_id_id= '8471';
select agency_id
from m.agency
where agency_id_id= '1803'
  and valid_now=17667;
select user_id
from m.agency
where valid_now=2838
  and agency_id_id= '12795';
select COUNT(*)
from dv.notes_message
where user_id='255'
  and agency_id_id= '255'
  and notice_id= '255'
  and route_id= '255';
select agency_id
from m.agency
where agency_id_id= '7756'
  and valid_now=2412;
select agency_id
from m.agency
where agency_id_id= '18640'
  and valid_now=3215;
select COUNT(*)
from dv.notes_message
where user_id='19737'
  and agency_id_id= '19737'
  and notice_id= '19737'
  and route_id= '19737';
select agency_id
from m.agency
where agency_id_id= '11377'
  and valid_now=17932;
select agency_id
from m.agency
where agency_id_id= '5943'
  and valid_now=6259;
select agency_id
from m.agency
where agency_id_id= '2509'
  and valid_now=8374;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14759'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14410'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16884'
  and valid_now=37;
select user_id
from m.agency
where valid_now=8504
  and agency_id_id= '6329';
select COUNT(*)
from dv.notes_message
where user_id='4057'
  and agency_id_id= '4057'
  and notice_id= '4057'
  and route_id= '4057';
select user_id
from m.agency
where valid_now=13577
  and agency_id_id= '18709';
select user_id
from m.agency
where valid_now=19317
  and agency_id_id= '8145';
select COUNT(*)
from dv.notes_message
where user_id='16649'
  and agency_id_id= '16649'
  and notice_id= '16649'
  and route_id= '16649';
select COUNT(*)
from dv.notes_message
where user_id='3139'
  and agency_id_id= '3139'
  and notice_id= '3139'
  and route_id= '3139';
select agency_id
from m.agency
where agency_id_id= '1842'
  and valid_now=13037;
select COUNT(*)
from dv.notes_message
where user_id='16228'
  and agency_id_id= '16228'
  and notice_id= '16228'
  and route_id= '16228';
select user_id
from m.agency
where valid_now=10568
  and agency_id_id= '15585';
select user_id
from m.agency
where valid_now=5683
  and agency_id_id= '17344';
select user_id
from m.agency
where valid_now=14287
  and agency_id_id= '14390';
select user_id
from m.agency
where valid_now=2802
  and agency_id_id= '6443';
select a.agency_timezone
from m.agency a
where a.agency_id = '10922';
select COUNT(*)
from dv.notes_message
where user_id='15818'
  and agency_id_id= '15818'
  and notice_id= '15818'
  and route_id= '15818';
select user_id
from m.agency
where valid_now=6952
  and agency_id_id= '5607';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19121'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '14199';
select user_id
from m.agency
where valid_now=2075
  and agency_id_id= '16349';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1470'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '12207';
select user_id
from m.agency
where valid_now=5526
  and agency_id_id= '5261';
select user_id
from m.agency
where valid_now=8482
  and agency_id_id= '19505';
select COUNT(*)
from dv.notes_message
where user_id='15801'
  and agency_id_id= '15801'
  and notice_id= '15801'
  and route_id= '15801';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18555'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11899'
  and agency_id_id= '11899'
  and notice_id= '11899'
  and route_id= '11899';
select agency_id
from m.agency
where agency_id_id= '9185'
  and valid_now=17338;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19395'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13433'
  and valid_now=848;
select user_id
from m.agency
where valid_now=19618
  and agency_id_id= '1993';
select COUNT(*)
from dv.notes_message
where user_id='16879'
  and agency_id_id= '16879'
  and notice_id= '16879'
  and route_id= '16879';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11706'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13408';
select a.agency_timezone
from m.agency a
where a.agency_id = '14795';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19010'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14306'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19624'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '12509';
select a.agency_timezone
from m.agency a
where a.agency_id = '9284';
select user_id
from m.agency
where valid_now=14270
  and agency_id_id= '8518';
select COUNT(*)
from dv.notes_message
where user_id='6133'
  and agency_id_id= '6133'
  and notice_id= '6133'
  and route_id= '6133';
select a.agency_timezone
from m.agency a
where a.agency_id = '13483';
select COUNT(*)
from dv.notes_message
where user_id='9606'
  and agency_id_id= '9606'
  and notice_id= '9606'
  and route_id= '9606';
select a.agency_timezone
from m.agency a
where a.agency_id = '2001';
select COUNT(*)
from dv.notes_message
where user_id='5366'
  and agency_id_id= '5366'
  and notice_id= '5366'
  and route_id= '5366';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13232'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4265'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=865
  and agency_id_id= '5139';
select COUNT(*)
from dv.notes_message
where user_id='12385'
  and agency_id_id= '12385'
  and notice_id= '12385'
  and route_id= '12385';
select COUNT(*)
from dv.notes_message
where user_id='12389'
  and agency_id_id= '12389'
  and notice_id= '12389'
  and route_id= '12389';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16215'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11946';
select user_id
from m.agency
where valid_now=16399
  and agency_id_id= '3237';
select user_id
from m.agency
where valid_now=6630
  and agency_id_id= '14039';
select COUNT(*)
from dv.notes_message
where user_id='13252'
  and agency_id_id= '13252'
  and notice_id= '13252'
  and route_id= '13252';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18080'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14573'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11507';
select a.agency_timezone
from m.agency a
where a.agency_id = '10572';
select user_id
from m.agency
where valid_now=6362
  and agency_id_id= '17156';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14990'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5439';
select a.agency_timezone
from m.agency a
where a.agency_id = '210';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16933'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1866'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14257'
  and valid_now=11497;
select COUNT(*)
from dv.notes_message
where user_id='154'
  and agency_id_id= '154'
  and notice_id= '154'
  and route_id= '154';
select COUNT(*)
from dv.notes_message
where user_id='7404'
  and agency_id_id= '7404'
  and notice_id= '7404'
  and route_id= '7404';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11897'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17087'
  and agency_id_id= '17087'
  and notice_id= '17087'
  and route_id= '17087';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10146'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12328'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '999'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9108
  and agency_id_id= '7975';
select user_id
from m.agency
where valid_now=1104
  and agency_id_id= '8257';
select user_id
from m.agency
where valid_now=18013
  and agency_id_id= '2517';
select agency_id
from m.agency
where agency_id_id= '8964'
  and valid_now=18294;
select agency_id
from m.agency
where agency_id_id= '13191'
  and valid_now=14366;
select user_id
from m.agency
where valid_now=6554
  and agency_id_id= '3114';
select user_id
from m.agency
where valid_now=4319
  and agency_id_id= '1225';
select COUNT(*)
from dv.notes_message
where user_id='9203'
  and agency_id_id= '9203'
  and notice_id= '9203'
  and route_id= '9203';
select COUNT(*)
from dv.notes_message
where user_id='18900'
  and agency_id_id= '18900'
  and notice_id= '18900'
  and route_id= '18900';
select agency_id
from m.agency
where agency_id_id= '6524'
  and valid_now=11844;
select user_id
from m.agency
where valid_now=4841
  and agency_id_id= '7828';
select user_id
from m.agency
where valid_now=15520
  and agency_id_id= '18722';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11430'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14902'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13311';
select agency_id
from m.agency
where agency_id_id= '17854'
  and valid_now=11303;
select COUNT(*)
from dv.notes_message
where user_id='8388'
  and agency_id_id= '8388'
  and notice_id= '8388'
  and route_id= '8388';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1395'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15830'
  and agency_id_id= '15830'
  and notice_id= '15830'
  and route_id= '15830';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17141'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16793'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13853';
select a.agency_timezone
from m.agency a
where a.agency_id = '16133';
select a.agency_timezone
from m.agency a
where a.agency_id = '3292';
select COUNT(*)
from dv.notes_message
where user_id='12118'
  and agency_id_id= '12118'
  and notice_id= '12118'
  and route_id= '12118';
select COUNT(*)
from dv.notes_message
where user_id='14182'
  and agency_id_id= '14182'
  and notice_id= '14182'
  and route_id= '14182';
select a.agency_timezone
from m.agency a
where a.agency_id = '5041';
select agency_id
from m.agency
where agency_id_id= '4068'
  and valid_now=15678;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18419'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18929'
  and valid_now=1104;
select COUNT(*)
from dv.notes_message
where user_id='16305'
  and agency_id_id= '16305'
  and notice_id= '16305'
  and route_id= '16305';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_1050'
  and t.trip_id = 19897
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_14161'
  and t.trip_id = 14217
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8104'
  and agency_id_id= '8104'
  and notice_id= '8104'
  and route_id= '8104';
select agency_id
from m.agency
where agency_id_id= '5503'
  and valid_now=15032;
select agency_id
from m.agency
where agency_id_id= '17427'
  and valid_now=3506;
select COUNT(*)
from dv.notes_message
where user_id='9103'
  and agency_id_id= '9103'
  and notice_id= '9103'
  and route_id= '9103';
select COUNT(*)
from dv.notes_message
where user_id='7313'
  and agency_id_id= '7313'
  and notice_id= '7313'
  and route_id= '7313';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_7889'
  and t.trip_id = 11223
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15639'
  and agency_id_id= '15639'
  and notice_id= '15639'
  and route_id= '15639';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_3545'
  and t.trip_id = 5796
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12194'
  and valid_now=942;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10328'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19396'
  and valid_now=9790;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3062'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12379'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11908'
  and valid_now=5668;
select user_id
from m.agency
where valid_now=1154
  and agency_id_id= '8283';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7542'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=147
  and agency_id_id= '12349';
select COUNT(*)
from dv.notes_message
where user_id='6374'
  and agency_id_id= '6374'
  and notice_id= '6374'
  and route_id= '6374';
select agency_id
from m.agency
where agency_id_id= '7203'
  and valid_now=5441;
select user_id
from m.agency
where valid_now=10635
  and agency_id_id= '141';
select COUNT(*)
from dv.notes_message
where user_id='17684'
  and agency_id_id= '17684'
  and notice_id= '17684'
  and route_id= '17684';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9211'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1856'
  and valid_now=5248;
select agency_id
from m.agency
where agency_id_id= '19520'
  and valid_now=18293;
select COUNT(*)
from dv.notes_message
where user_id='14708'
  and agency_id_id= '14708'
  and notice_id= '14708'
  and route_id= '14708';
select agency_id
from m.agency
where agency_id_id= '6149'
  and valid_now=7488;
select agency_id
from m.agency
where agency_id_id= '6408'
  and valid_now=8472;
select user_id
from m.agency
where valid_now=7433
  and agency_id_id= '17262';
select COUNT(*)
from dv.notes_message
where user_id='5940'
  and agency_id_id= '5940'
  and notice_id= '5940'
  and route_id= '5940';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17790'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4733'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16700'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6664'
  and valid_now=3604;
select agency_id
from m.agency
where agency_id_id= '2017'
  and valid_now=7115;
select user_id
from m.agency
where valid_now=19551
  and agency_id_id= '13038';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6392'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15170'
  and valid_now=5198;
select COUNT(*)
from dv.notes_message
where user_id='16802'
  and agency_id_id= '16802'
  and notice_id= '16802'
  and route_id= '16802';
select COUNT(*)
from dv.notes_message
where user_id='18519'
  and agency_id_id= '18519'
  and notice_id= '18519'
  and route_id= '18519';
select COUNT(*)
from dv.notes_message
where user_id='5422'
  and agency_id_id= '5422'
  and notice_id= '5422'
  and route_id= '5422';
select a.agency_timezone
from m.agency a
where a.agency_id = '15284';
select a.agency_timezone
from m.agency a
where a.agency_id = '17769';
select COUNT(*)
from dv.notes_message
where user_id='18243'
  and agency_id_id= '18243'
  and notice_id= '18243'
  and route_id= '18243';
select a.agency_timezone
from m.agency a
where a.agency_id = '13281';
select a.agency_timezone
from m.agency a
where a.agency_id = '7888';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10020'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '7913';
select a.agency_timezone
from m.agency a
where a.agency_id = '8515';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5062'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1567'
  and valid_now=10412;
select agency_id
from m.agency
where agency_id_id= '14504'
  and valid_now=19801;
select COUNT(*)
from dv.notes_message
where user_id='7162'
  and agency_id_id= '7162'
  and notice_id= '7162'
  and route_id= '7162';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '994'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3339'
  and valid_now=12932;
select agency_id
from m.agency
where agency_id_id= '4086'
  and valid_now=18682;
select agency_id
from m.agency
where agency_id_id= '18266'
  and valid_now=12026;
select agency_id
from m.agency
where agency_id_id= '4191'
  and valid_now=12207;
select COUNT(*)
from dv.notes_message
where user_id='6157'
  and agency_id_id= '6157'
  and notice_id= '6157'
  and route_id= '6157';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8553'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9562'
  and agency_id_id= '9562'
  and notice_id= '9562'
  and route_id= '9562';
select COUNT(*)
from dv.notes_message
where user_id='7026'
  and agency_id_id= '7026'
  and notice_id= '7026'
  and route_id= '7026';
select COUNT(*)
from dv.notes_message
where user_id='16527'
  and agency_id_id= '16527'
  and notice_id= '16527'
  and route_id= '16527';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7712'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17633'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=830
  and agency_id_id= '536';
select COUNT(*)
from dv.notes_message
where user_id='13487'
  and agency_id_id= '13487'
  and notice_id= '13487'
  and route_id= '13487';
select user_id
from m.agency
where valid_now=1623
  and agency_id_id= '10365';
select user_id
from m.agency
where valid_now=4272
  and agency_id_id= '19894';
select user_id
from m.agency
where valid_now=794
  and agency_id_id= '7779';
select COUNT(*)
from dv.notes_message
where user_id='938'
  and agency_id_id= '938'
  and notice_id= '938'
  and route_id= '938';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8868'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '371'
  and valid_now=11982;
select agency_id
from m.agency
where agency_id_id= '10704'
  and valid_now=9253;
select COUNT(*)
from dv.notes_message
where user_id='3897'
  and agency_id_id= '3897'
  and notice_id= '3897'
  and route_id= '3897';
select COUNT(*)
from dv.notes_message
where user_id='5606'
  and agency_id_id= '5606'
  and notice_id= '5606'
  and route_id= '5606';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2356'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1784'
  and valid_now=10148;
select agency_id
from m.agency
where agency_id_id= '3618'
  and valid_now=18877;
select COUNT(*)
from dv.notes_message
where user_id='5839'
  and agency_id_id= '5839'
  and notice_id= '5839'
  and route_id= '5839';
select a.agency_timezone
from m.agency a
where a.agency_id = '4654';
select agency_id
from m.agency
where agency_id_id= '10554'
  and valid_now=4977;
select agency_id
from m.agency
where agency_id_id= '15414'
  and valid_now=17710;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3004'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11492'
  and agency_id_id= '11492'
  and notice_id= '11492'
  and route_id= '11492';
select agency_id
from m.agency
where agency_id_id= '9866'
  and valid_now=1413;
select agency_id
from m.agency
where agency_id_id= '1934'
  and valid_now=2289;
select COUNT(*)
from dv.notes_message
where user_id='5572'
  and agency_id_id= '5572'
  and notice_id= '5572'
  and route_id= '5572';
select agency_id
from m.agency
where agency_id_id= '10138'
  and valid_now=8885;
select COUNT(*)
from dv.notes_message
where user_id='18591'
  and agency_id_id= '18591'
  and notice_id= '18591'
  and route_id= '18591';
select COUNT(*)
from dv.notes_message
where user_id='19150'
  and agency_id_id= '19150'
  and notice_id= '19150'
  and route_id= '19150';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15206'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7733'
  and valid_now=314;
select agency_id
from m.agency
where agency_id_id= '8163'
  and valid_now=13072;
select COUNT(*)
from dv.notes_message
where user_id='8246'
  and agency_id_id= '8246'
  and notice_id= '8246'
  and route_id= '8246';
select a.agency_timezone
from m.agency a
where a.agency_id = '7469';
select agency_id
from m.agency
where agency_id_id= '18106'
  and valid_now=4339;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18326'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '15740';
select agency_id
from m.agency
where agency_id_id= '3478'
  and valid_now=9197;
select agency_id
from m.agency
where agency_id_id= '9069'
  and valid_now=7602;
select COUNT(*)
from dv.notes_message
where user_id='11577'
  and agency_id_id= '11577'
  and notice_id= '11577'
  and route_id= '11577';
select agency_id
from m.agency
where agency_id_id= '12613'
  and valid_now=15539;
select COUNT(*)
from dv.notes_message
where user_id='13361'
  and agency_id_id= '13361'
  and notice_id= '13361'
  and route_id= '13361';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14524'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8764'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14128'
  and agency_id_id= '14128'
  and notice_id= '14128'
  and route_id= '14128';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3719'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14178'
  and valid_now=18733;
select COUNT(*)
from dv.notes_message
where user_id='17651'
  and agency_id_id= '17651'
  and notice_id= '17651'
  and route_id= '17651';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19494'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6941'
  and valid_now=1481;
select user_id
from m.agency
where valid_now=11231
  and agency_id_id= '1566';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15155'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10395'
  and valid_now=12055;
select user_id
from m.agency
where valid_now=19955
  and agency_id_id= '17721';
select COUNT(*)
from dv.notes_message
where user_id='2676'
  and agency_id_id= '2676'
  and notice_id= '2676'
  and route_id= '2676';
select COUNT(*)
from dv.notes_message
where user_id='8633'
  and agency_id_id= '8633'
  and notice_id= '8633'
  and route_id= '8633';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14522'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10351'
  and valid_now=17713;
select user_id
from m.agency
where valid_now=13104
  and agency_id_id= '5022';
select COUNT(*)
from dv.notes_message
where user_id='9322'
  and agency_id_id= '9322'
  and notice_id= '9322'
  and route_id= '9322';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18117'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1665'
  and valid_now=2478;
select user_id
from m.agency
where valid_now=3018
  and agency_id_id= '1974';
select agency_id
from m.agency
where agency_id_id= '18991'
  and valid_now=2916;
select user_id
from m.agency
where valid_now=14137
  and agency_id_id= '14276';
select user_id
from m.agency
where valid_now=6849
  and agency_id_id= '3868';
select agency_id
from m.agency
where agency_id_id= '3713'
  and valid_now=784;
select COUNT(*)
from dv.notes_message
where user_id='16380'
  and agency_id_id= '16380'
  and notice_id= '16380'
  and route_id= '16380';
select COUNT(*)
from dv.notes_message
where user_id='16194'
  and agency_id_id= '16194'
  and notice_id= '16194'
  and route_id= '16194';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15313'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2707'
  and agency_id_id= '2707'
  and notice_id= '2707'
  and route_id= '2707';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3355'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11143'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15031'
  and valid_now=16162;
select agency_id
from m.agency
where agency_id_id= '11347'
  and valid_now=17065;
select user_id
from m.agency
where valid_now=18979
  and agency_id_id= '2585';
select agency_id
from m.agency
where agency_id_id= '8821'
  and valid_now=18939;
select user_id
from m.agency
where valid_now=4634
  and agency_id_id= '17112';
select COUNT(*)
from dv.notes_message
where user_id='2138'
  and agency_id_id= '2138'
  and notice_id= '2138'
  and route_id= '2138';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13890'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10064'
  and valid_now=4170;
select user_id
from m.agency
where valid_now=18143
  and agency_id_id= '2843';
select COUNT(*)
from dv.notes_message
where user_id='15376'
  and agency_id_id= '15376'
  and notice_id= '15376'
  and route_id= '15376';
select user_id
from m.agency
where valid_now=9792
  and agency_id_id= '5310';
select COUNT(*)
from dv.notes_message
where user_id='2596'
  and agency_id_id= '2596'
  and notice_id= '2596'
  and route_id= '2596';
select COUNT(*)
from dv.notes_message
where user_id='16058'
  and agency_id_id= '16058'
  and notice_id= '16058'
  and route_id= '16058';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10226'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3222'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15483
  and agency_id_id= '6939';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17637'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15550
  and agency_id_id= '17831';
select agency_id
from m.agency
where agency_id_id= '19745'
  and valid_now=2458;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9182'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19884
  and agency_id_id= '10797';
select user_id
from m.agency
where valid_now=12813
  and agency_id_id= '8666';
select COUNT(*)
from dv.notes_message
where user_id='14133'
  and agency_id_id= '14133'
  and notice_id= '14133'
  and route_id= '14133';
select COUNT(*)
from dv.notes_message
where user_id='17187'
  and agency_id_id= '17187'
  and notice_id= '17187'
  and route_id= '17187';
select COUNT(*)
from dv.notes_message
where user_id='11817'
  and agency_id_id= '11817'
  and notice_id= '11817'
  and route_id= '11817';
select agency_id
from m.agency
where agency_id_id= '570'
  and valid_now=16954;
select agency_id
from m.agency
where agency_id_id= '9952'
  and valid_now=6215;
select user_id
from m.agency
where valid_now=6003
  and agency_id_id= '10691';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1155'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5166'
  and valid_now=14216;
select agency_id
from m.agency
where agency_id_id= '16528'
  and valid_now=14299;
select COUNT(*)
from dv.notes_message
where user_id='5931'
  and agency_id_id= '5931'
  and notice_id= '5931'
  and route_id= '5931';
select COUNT(*)
from dv.notes_message
where user_id='17763'
  and agency_id_id= '17763'
  and notice_id= '17763'
  and route_id= '17763';
select agency_id
from m.agency
where agency_id_id= '12112'
  and valid_now=19471;
select COUNT(*)
from dv.notes_message
where user_id='18306'
  and agency_id_id= '18306'
  and notice_id= '18306'
  and route_id= '18306';
select COUNT(*)
from dv.notes_message
where user_id='9795'
  and agency_id_id= '9795'
  and notice_id= '9795'
  and route_id= '9795';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8825'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '279'
  and valid_now=18883;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4568'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15760
  and agency_id_id= '16626';
select COUNT(*)
from dv.notes_message
where user_id='6686'
  and agency_id_id= '6686'
  and notice_id= '6686'
  and route_id= '6686';
select COUNT(*)
from dv.notes_message
where user_id='724'
  and agency_id_id= '724'
  and notice_id= '724'
  and route_id= '724';
select COUNT(*)
from dv.notes_message
where user_id='4279'
  and agency_id_id= '4279'
  and notice_id= '4279'
  and route_id= '4279';
select agency_id
from m.agency
where agency_id_id= '17007'
  and valid_now=17990;
select agency_id
from m.agency
where agency_id_id= '8281'
  and valid_now=7309;
select COUNT(*)
from dv.notes_message
where user_id='14109'
  and agency_id_id= '14109'
  and notice_id= '14109'
  and route_id= '14109';
select agency_id
from m.agency
where agency_id_id= '15833'
  and valid_now=4078;
select COUNT(*)
from dv.notes_message
where user_id='14158'
  and agency_id_id= '14158'
  and notice_id= '14158'
  and route_id= '14158';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11624'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3219'
  and valid_now=2119;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4164'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6441'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14638'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2043'
  and valid_now=5059;
select user_id
from m.agency
where valid_now=12899
  and agency_id_id= '2169';
select COUNT(*)
from dv.notes_message
where user_id='1799'
  and agency_id_id= '1799'
  and notice_id= '1799'
  and route_id= '1799';
select COUNT(*)
from dv.notes_message
where user_id='415'
  and agency_id_id= '415'
  and notice_id= '415'
  and route_id= '415';
select COUNT(*)
from dv.notes_message
where user_id='16378'
  and agency_id_id= '16378'
  and notice_id= '16378'
  and route_id= '16378';
select COUNT(*)
from dv.notes_message
where user_id='14711'
  and agency_id_id= '14711'
  and notice_id= '14711'
  and route_id= '14711';
select agency_id
from m.agency
where agency_id_id= '10160'
  and valid_now=4636;
select user_id
from m.agency
where valid_now=13680
  and agency_id_id= '2757';
select COUNT(*)
from dv.notes_message
where user_id='17260'
  and agency_id_id= '17260'
  and notice_id= '17260'
  and route_id= '17260';
select COUNT(*)
from dv.notes_message
where user_id='9578'
  and agency_id_id= '9578'
  and notice_id= '9578'
  and route_id= '9578';
select COUNT(*)
from dv.notes_message
where user_id='13387'
  and agency_id_id= '13387'
  and notice_id= '13387'
  and route_id= '13387';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12372'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4844'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2282'
  and valid_now=10942;
select a.agency_timezone
from m.agency a
where a.agency_id = '18265';
select agency_id
from m.agency
where agency_id_id= '7396'
  and valid_now=10785;
select agency_id
from m.agency
where agency_id_id= '13771'
  and valid_now=9630;
select agency_id
from m.agency
where agency_id_id= '665'
  and valid_now=13809;
select COUNT(*)
from dv.notes_message
where user_id='14747'
  and agency_id_id= '14747'
  and notice_id= '14747'
  and route_id= '14747';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17600'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '16536';
select agency_id
from m.agency
where agency_id_id= '10118'
  and valid_now=1728;
select agency_id
from m.agency
where agency_id_id= '10386'
  and valid_now=932;
select COUNT(*)
from dv.notes_message
where user_id='11841'
  and agency_id_id= '11841'
  and notice_id= '11841'
  and route_id= '11841';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8211'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2363'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2046'
  and agency_id_id= '2046'
  and notice_id= '2046'
  and route_id= '2046';
select COUNT(*)
from dv.notes_message
where user_id='293'
  and agency_id_id= '293'
  and notice_id= '293'
  and route_id= '293';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3348'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '2595';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7408'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6011'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='10444'
  and agency_id_id= '10444'
  and notice_id= '10444'
  and route_id= '10444';
select a.agency_timezone
from m.agency a
where a.agency_id = '11637';
select a.agency_timezone
from m.agency a
where a.agency_id = '18881';
select a.agency_timezone
from m.agency a
where a.agency_id = '6543';
select a.agency_timezone
from m.agency a
where a.agency_id = '3824';
select a.agency_timezone
from m.agency a
where a.agency_id = '1740';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17983'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15591'
  and agency_id_id= '15591'
  and notice_id= '15591'
  and route_id= '15591';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15419'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '14944';
select COUNT(*)
from dv.notes_message
where user_id='4360'
  and agency_id_id= '4360'
  and notice_id= '4360'
  and route_id= '4360';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10703'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18588';
select COUNT(*)
from dv.notes_message
where user_id='12049'
  and agency_id_id= '12049'
  and notice_id= '12049'
  and route_id= '12049';
select COUNT(*)
from dv.notes_message
where user_id='2411'
  and agency_id_id= '2411'
  and notice_id= '2411'
  and route_id= '2411';
select agency_id
from m.agency
where agency_id_id= '3127'
  and valid_now=8772;
select agency_id
from m.agency
where agency_id_id= '9547'
  and valid_now=9877;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7910'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '14994';
select COUNT(*)
from dv.notes_message
where user_id='9309'
  and agency_id_id= '9309'
  and notice_id= '9309'
  and route_id= '9309';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5210'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '535'
  and valid_now=16480;
select agency_id
from m.agency
where agency_id_id= '2281'
  and valid_now=8689;
select agency_id
from m.agency
where agency_id_id= '18862'
  and valid_now=10299;
select COUNT(*)
from dv.notes_message
where user_id='2265'
  and agency_id_id= '2265'
  and notice_id= '2265'
  and route_id= '2265';
select COUNT(*)
from dv.notes_message
where user_id='13805'
  and agency_id_id= '13805'
  and notice_id= '13805'
  and route_id= '13805';
select agency_id
from m.agency
where agency_id_id= '18443'
  and valid_now=17237;
select agency_id
from m.agency
where agency_id_id= '1533'
  and valid_now=13521;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5833'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14808'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7867
  and agency_id_id= '9385';
select COUNT(*)
from dv.notes_message
where user_id='10719'
  and agency_id_id= '10719'
  and notice_id= '10719'
  and route_id= '10719';
select agency_id
from m.agency
where agency_id_id= '19509'
  and valid_now=10240;
select user_id
from m.agency
where valid_now=6824
  and agency_id_id= '15672';
select COUNT(*)
from dv.notes_message
where user_id='12776'
  and agency_id_id= '12776'
  and notice_id= '12776'
  and route_id= '12776';
select agency_id
from m.agency
where agency_id_id= '16478'
  and valid_now=12114;
select agency_id
from m.agency
where agency_id_id= '11343'
  and valid_now=7777;
select user_id
from m.agency
where valid_now=10254
  and agency_id_id= '19250';
select COUNT(*)
from dv.notes_message
where user_id='7137'
  and agency_id_id= '7137'
  and notice_id= '7137'
  and route_id= '7137';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19524'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14226'
  and valid_now=11270;
select agency_id
from m.agency
where agency_id_id= '13222'
  and valid_now=18121;
select agency_id
from m.agency
where agency_id_id= '6856'
  and valid_now=7568;
select agency_id
from m.agency
where agency_id_id= '15689'
  and valid_now=18708;
select user_id
from m.agency
where valid_now=2835
  and agency_id_id= '4831';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12845'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11171'
  and agency_id_id= '11171'
  and notice_id= '11171'
  and route_id= '11171';
select COUNT(*)
from dv.notes_message
where user_id='7409'
  and agency_id_id= '7409'
  and notice_id= '7409'
  and route_id= '7409';
select COUNT(*)
from dv.notes_message
where user_id='10902'
  and agency_id_id= '10902'
  and notice_id= '10902'
  and route_id= '10902';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11481'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15972'
  and valid_now=5276;
select user_id
from m.agency
where valid_now=14871
  and agency_id_id= '5775';
select COUNT(*)
from dv.notes_message
where user_id='4642'
  and agency_id_id= '4642'
  and notice_id= '4642'
  and route_id= '4642';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5665'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4731'
  and valid_now=16647;
select agency_id
from m.agency
where agency_id_id= '6892'
  and valid_now=15445;
select user_id
from m.agency
where valid_now=12105
  and agency_id_id= '1518';
select COUNT(*)
from dv.notes_message
where user_id='1851'
  and agency_id_id= '1851'
  and notice_id= '1851'
  and route_id= '1851';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9178'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19887
  and agency_id_id= '10329';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15450'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3611'
  and valid_now=13541;
select agency_id
from m.agency
where agency_id_id= '12717'
  and valid_now=4076;
select COUNT(*)
from dv.notes_message
where user_id='10775'
  and agency_id_id= '10775'
  and notice_id= '10775'
  and route_id= '10775';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_12940'
  and t.trip_id = 2624
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_11163'
  and t.trip_id = 18820
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12341'
  and agency_id_id= '12341'
  and notice_id= '12341'
  and route_id= '12341';
select agency_id
from m.agency
where agency_id_id= '14478'
  and valid_now=13437;
select COUNT(*)
from dv.notes_message
where user_id='18921'
  and agency_id_id= '18921'
  and notice_id= '18921'
  and route_id= '18921';
select COUNT(*)
from dv.notes_message
where user_id='13055'
  and agency_id_id= '13055'
  and notice_id= '13055'
  and route_id= '13055';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18311'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14441'
  and valid_now=16612;
select user_id
from m.agency
where valid_now=17805
  and agency_id_id= '7227';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14949'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6297'
  and valid_now=19924;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8362'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4417'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13386
  and agency_id_id= '13143';
select user_id
from m.agency
where valid_now=3374
  and agency_id_id= '4963';
select agency_id
from m.agency
where agency_id_id= '16377'
  and valid_now=9724;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12305'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1696'
  and valid_now=17111;
select user_id
from m.agency
where valid_now=8274
  and agency_id_id= '18883';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9676'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6673'
  and valid_now=14756;
select user_id
from m.agency
where valid_now=3497
  and agency_id_id= '13521';
select COUNT(*)
from dv.notes_message
where user_id='17687'
  and agency_id_id= '17687'
  and notice_id= '17687'
  and route_id= '17687';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11155'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6936'
  and valid_now=8103;
select user_id
from m.agency
where valid_now=2083
  and agency_id_id= '19829';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19071'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1216'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6069'
  and valid_now=8181;
select COUNT(*)
from dv.notes_message
where user_id='17323'
  and agency_id_id= '17323'
  and notice_id= '17323'
  and route_id= '17323';
select user_id
from m.agency
where valid_now=16518
  and agency_id_id= '5767';
select COUNT(*)
from dv.notes_message
where user_id='16193'
  and agency_id_id= '16193'
  and notice_id= '16193'
  and route_id= '16193';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3745'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2737'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8521'
  and agency_id_id= '8521'
  and notice_id= '8521'
  and route_id= '8521';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18329
  and agency_id_id= '9369';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7102'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7505'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18390
  and agency_id_id= '16657';
select agency_id
from m.agency
where agency_id_id= '13955'
  and valid_now=18913;
select agency_id
from m.agency
where agency_id_id= '283'
  and valid_now=8335;
select user_id
from m.agency
where valid_now=16858
  and agency_id_id= '18519';
select COUNT(*)
from dv.notes_message
where user_id='5427'
  and agency_id_id= '5427'
  and notice_id= '5427'
  and route_id= '5427';
select agency_id
from m.agency
where agency_id_id= '15130'
  and valid_now=10427;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10856'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5024'
  and valid_now=17916;
select agency_id
from m.agency
where agency_id_id= '5575'
  and valid_now=133;
select user_id
from m.agency
where valid_now=3728
  and agency_id_id= '17174';
select agency_id
from m.agency
where agency_id_id= '17131'
  and valid_now=3851;
select COUNT(*)
from dv.notes_message
where user_id='15944'
  and agency_id_id= '15944'
  and notice_id= '15944'
  and route_id= '15944';
select COUNT(*)
from dv.notes_message
where user_id='15044'
  and agency_id_id= '15044'
  and notice_id= '15044'
  and route_id= '15044';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '920'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '692'
  and valid_now=8901;
select COUNT(*)
from dv.notes_message
where user_id='17294'
  and agency_id_id= '17294'
  and notice_id= '17294'
  and route_id= '17294';
select COUNT(*)
from dv.notes_message
where user_id='3082'
  and agency_id_id= '3082'
  and notice_id= '3082'
  and route_id= '3082';
select user_id
from m.agency
where valid_now=4987
  and agency_id_id= '7267';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19350'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18607
  and agency_id_id= '4801';
select COUNT(*)
from dv.notes_message
where user_id='13059'
  and agency_id_id= '13059'
  and notice_id= '13059'
  and route_id= '13059';
select agency_id
from m.agency
where agency_id_id= '14585'
  and valid_now=10509;
select COUNT(*)
from dv.notes_message
where user_id='1308'
  and agency_id_id= '1308'
  and notice_id= '1308'
  and route_id= '1308';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9508'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18874'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2065'
  and valid_now=6322;
select COUNT(*)
from dv.notes_message
where user_id='16726'
  and agency_id_id= '16726'
  and notice_id= '16726'
  and route_id= '16726';
select agency_id
from m.agency
where agency_id_id= '13767'
  and valid_now=1926;
select agency_id
from m.agency
where agency_id_id= '4283'
  and valid_now=9791;
select agency_id
from m.agency
where agency_id_id= '13983'
  and valid_now=7274;
select agency_id
from m.agency
where agency_id_id= '4813'
  and valid_now=8218;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18013'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2173'
  and agency_id_id= '2173'
  and notice_id= '2173'
  and route_id= '2173';
select agency_id
from m.agency
where agency_id_id= '15338'
  and valid_now=16762;
select COUNT(*)
from dv.notes_message
where user_id='18265'
  and agency_id_id= '18265'
  and notice_id= '18265'
  and route_id= '18265';
select COUNT(*)
from dv.notes_message
where user_id='6309'
  and agency_id_id= '6309'
  and notice_id= '6309'
  and route_id= '6309';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4099'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4322'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9201'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11785'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18022
  and agency_id_id= '11354';
select user_id
from m.agency
where valid_now=10819
  and agency_id_id= '10422';
select COUNT(*)
from dv.notes_message
where user_id='16935'
  and agency_id_id= '16935'
  and notice_id= '16935'
  and route_id= '16935';
select agency_id
from m.agency
where agency_id_id= '2423'
  and valid_now=4108;
select agency_id
from m.agency
where agency_id_id= '8447'
  and valid_now=2326;
select user_id
from m.agency
where valid_now=12955
  and agency_id_id= '4092';
select user_id
from m.agency
where valid_now=4631
  and agency_id_id= '11781';
select COUNT(*)
from dv.notes_message
where user_id='6193'
  and agency_id_id= '6193'
  and notice_id= '6193'
  and route_id= '6193';
select agency_id
from m.agency
where agency_id_id= '14401'
  and valid_now=19637;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13854'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7309'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15761'
  and valid_now=4259;
select COUNT(*)
from dv.notes_message
where user_id='1679'
  and agency_id_id= '1679'
  and notice_id= '1679'
  and route_id= '1679';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7439'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=413
  and agency_id_id= '5572';
select COUNT(*)
from dv.notes_message
where user_id='8526'
  and agency_id_id= '8526'
  and notice_id= '8526'
  and route_id= '8526';
select COUNT(*)
from dv.notes_message
where user_id='10386'
  and agency_id_id= '10386'
  and notice_id= '10386'
  and route_id= '10386';
select COUNT(*)
from dv.notes_message
where user_id='4224'
  and agency_id_id= '4224'
  and notice_id= '4224'
  and route_id= '4224';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14036'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18536'
  and agency_id_id= '18536'
  and notice_id= '18536'
  and route_id= '18536';
select user_id
from m.agency
where valid_now=11353
  and agency_id_id= '13097';
select COUNT(*)
from dv.notes_message
where user_id='9253'
  and agency_id_id= '9253'
  and notice_id= '9253'
  and route_id= '9253';
select user_id
from m.agency
where valid_now=3317
  and agency_id_id= '3663';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8201'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3334
  and agency_id_id= '1568';
select COUNT(*)
from dv.notes_message
where user_id='6583'
  and agency_id_id= '6583'
  and notice_id= '6583'
  and route_id= '6583';
select agency_id
from m.agency
where agency_id_id= '11742'
  and valid_now=16063;
select COUNT(*)
from dv.notes_message
where user_id='19738'
  and agency_id_id= '19738'
  and notice_id= '19738'
  and route_id= '19738';
select COUNT(*)
from dv.notes_message
where user_id='15581'
  and agency_id_id= '15581'
  and notice_id= '15581'
  and route_id= '15581';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14298'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2947'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4112'
  and valid_now=7184;
select agency_id
from m.agency
where agency_id_id= '3594'
  and valid_now=11613;
select COUNT(*)
from dv.notes_message
where user_id='11019'
  and agency_id_id= '11019'
  and notice_id= '11019'
  and route_id= '11019';
select agency_id
from m.agency
where agency_id_id= '6414'
  and valid_now=7050;
select COUNT(*)
from dv.notes_message
where user_id='261'
  and agency_id_id= '261'
  and notice_id= '261'
  and route_id= '261';
select agency_id
from m.agency
where agency_id_id= '13069'
  and valid_now=11534;
select agency_id
from m.agency
where agency_id_id= '5697'
  and valid_now=14521;
select agency_id
from m.agency
where agency_id_id= '2176'
  and valid_now=15749;
select agency_id
from m.agency
where agency_id_id= '17395'
  and valid_now=19673;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12741'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18334';
select a.agency_timezone
from m.agency a
where a.agency_id = '9059';
select a.agency_timezone
from m.agency a
where a.agency_id = '3392';
select a.agency_timezone
from m.agency a
where a.agency_id = '19459';
select agency_id
from m.agency
where agency_id_id= '4778'
  and valid_now=823;
select user_id
from m.agency
where valid_now=775
  and agency_id_id= '8464';
select user_id
from m.agency
where valid_now=5248
  and agency_id_id= '8814';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10409'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14020'
  and valid_now=8472;
select COUNT(*)
from dv.notes_message
where user_id='17061'
  and agency_id_id= '17061'
  and notice_id= '17061'
  and route_id= '17061';
select agency_id
from m.agency
where agency_id_id= '7158'
  and valid_now=8756;
select user_id
from m.agency
where valid_now=19083
  and agency_id_id= '4146';
select user_id
from m.agency
where valid_now=14886
  and agency_id_id= '8145';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8802'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '904';
select agency_id
from m.agency
where agency_id_id= '4468'
  and valid_now=1143;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18526'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6622'
  and valid_now=8589;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1743'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18221'
  and agency_id_id= '18221'
  and notice_id= '18221'
  and route_id= '18221';
select COUNT(*)
from dv.notes_message
where user_id='11053'
  and agency_id_id= '11053'
  and notice_id= '11053'
  and route_id= '11053';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16431'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '16585';
select agency_id
from m.agency
where agency_id_id= '18941'
  and valid_now=13183;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1499'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '732';
select agency_id
from m.agency
where agency_id_id= '17279'
  and valid_now=8364;
select COUNT(*)
from dv.notes_message
where user_id='6316'
  and agency_id_id= '6316'
  and notice_id= '6316'
  and route_id= '6316';
select a.agency_timezone
from m.agency a
where a.agency_id = '1397';
select COUNT(*)
from dv.notes_message
where user_id='15708'
  and agency_id_id= '15708'
  and notice_id= '15708'
  and route_id= '15708';
select a.agency_timezone
from m.agency a
where a.agency_id = '555';
select COUNT(*)
from dv.notes_message
where user_id='4334'
  and agency_id_id= '4334'
  and notice_id= '4334'
  and route_id= '4334';
select a.agency_timezone
from m.agency a
where a.agency_id = '367';
select a.agency_timezone
from m.agency a
where a.agency_id = '18026';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2920'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14220'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2426'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13398'
  and agency_id_id= '13398'
  and notice_id= '13398'
  and route_id= '13398';
select COUNT(*)
from dv.notes_message
where user_id='129'
  and agency_id_id= '129'
  and notice_id= '129'
  and route_id= '129';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2847'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2737'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15020'
  and valid_now=2476;
select COUNT(*)
from dv.notes_message
where user_id='18394'
  and agency_id_id= '18394'
  and notice_id= '18394'
  and route_id= '18394';
select COUNT(*)
from dv.notes_message
where user_id='13647'
  and agency_id_id= '13647'
  and notice_id= '13647'
  and route_id= '13647';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16493'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3864'
  and agency_id_id= '3864'
  and notice_id= '3864'
  and route_id= '3864';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9960'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19436
  and agency_id_id= '15243';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12517'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6719'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11765'
  and valid_now=17900;
select agency_id
from m.agency
where agency_id_id= '304'
  and valid_now=17923;
select COUNT(*)
from dv.notes_message
where user_id='13303'
  and agency_id_id= '13303'
  and notice_id= '13303'
  and route_id= '13303';
select COUNT(*)
from dv.notes_message
where user_id='19685'
  and agency_id_id= '19685'
  and notice_id= '19685'
  and route_id= '19685';
select agency_id
from m.agency
where agency_id_id= '14289'
  and valid_now=3350;
select COUNT(*)
from dv.notes_message
where user_id='19042'
  and agency_id_id= '19042'
  and notice_id= '19042'
  and route_id= '19042';
select COUNT(*)
from dv.notes_message
where user_id='4489'
  and agency_id_id= '4489'
  and notice_id= '4489'
  and route_id= '4489';
select COUNT(*)
from dv.notes_message
where user_id='2917'
  and agency_id_id= '2917'
  and notice_id= '2917'
  and route_id= '2917';
select a.agency_timezone
from m.agency a
where a.agency_id = '16360';
select agency_id
from m.agency
where agency_id_id= '13417'
  and valid_now=14195;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14922'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8056'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1036'
  and valid_now=3619;
select user_id
from m.agency
where valid_now=7621
  and agency_id_id= '19537';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '356'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2582'
  and valid_now=17645;
select user_id
from m.agency
where valid_now=9827
  and agency_id_id= '15546';
select COUNT(*)
from dv.notes_message
where user_id='8466'
  and agency_id_id= '8466'
  and notice_id= '8466'
  and route_id= '8466';
select agency_id
from m.agency
where agency_id_id= '12782'
  and valid_now=14676;
select COUNT(*)
from dv.notes_message
where user_id='4264'
  and agency_id_id= '4264'
  and notice_id= '4264'
  and route_id= '4264';
select COUNT(*)
from dv.notes_message
where user_id='18113'
  and agency_id_id= '18113'
  and notice_id= '18113'
  and route_id= '18113';
select COUNT(*)
from dv.notes_message
where user_id='5998'
  and agency_id_id= '5998'
  and notice_id= '5998'
  and route_id= '5998';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12279'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1907
  and agency_id_id= '15733';
select COUNT(*)
from dv.notes_message
where user_id='7293'
  and agency_id_id= '7293'
  and notice_id= '7293'
  and route_id= '7293';
select a.agency_timezone
from m.agency a
where a.agency_id = '8318';
select agency_id
from m.agency
where agency_id_id= '1779'
  and valid_now=9404;
select agency_id
from m.agency
where agency_id_id= '13339'
  and valid_now=17081;
select agency_id
from m.agency
where agency_id_id= '5206'
  and valid_now=1864;
select user_id
from m.agency
where valid_now=12639
  and agency_id_id= '5590';
select user_id
from m.agency
where valid_now=1695
  and agency_id_id= '16803';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7310'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9660
  and agency_id_id= '11841';
select user_id
from m.agency
where valid_now=7690
  and agency_id_id= '18192';
select agency_id
from m.agency
where agency_id_id= '3448'
  and valid_now=15498;
select user_id
from m.agency
where valid_now=11423
  and agency_id_id= '13744';
select user_id
from m.agency
where valid_now=8165
  and agency_id_id= '9142';
select COUNT(*)
from dv.notes_message
where user_id='6410'
  and agency_id_id= '6410'
  and notice_id= '6410'
  and route_id= '6410';
select user_id
from m.agency
where valid_now=4733
  and agency_id_id= '10035';
select user_id
from m.agency
where valid_now=5860
  and agency_id_id= '15925';
select user_id
from m.agency
where valid_now=8613
  and agency_id_id= '14855';
select user_id
from m.agency
where valid_now=2723
  and agency_id_id= '3921';
select agency_id
from m.agency
where agency_id_id= '18308'
  and valid_now=7420;
select COUNT(*)
from dv.notes_message
where user_id='6911'
  and agency_id_id= '6911'
  and notice_id= '6911'
  and route_id= '6911';
select agency_id
from m.agency
where agency_id_id= '9069'
  and valid_now=4377;
select agency_id
from m.agency
where agency_id_id= '17843'
  and valid_now=4804;
select COUNT(*)
from dv.notes_message
where user_id='17778'
  and agency_id_id= '17778'
  and notice_id= '17778'
  and route_id= '17778';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16810'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6188'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14311
  and agency_id_id= '16344';
select user_id
from m.agency
where valid_now=3900
  and agency_id_id= '17316';
select COUNT(*)
from dv.notes_message
where user_id='2110'
  and agency_id_id= '2110'
  and notice_id= '2110'
  and route_id= '2110';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18084'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11362'
  and valid_now=10374;
select agency_id
from m.agency
where agency_id_id= '15671'
  and valid_now=14644;
select user_id
from m.agency
where valid_now=11758
  and agency_id_id= '5964';
select user_id
from m.agency
where valid_now=13948
  and agency_id_id= '6161';
select COUNT(*)
from dv.notes_message
where user_id='12032'
  and agency_id_id= '12032'
  and notice_id= '12032'
  and route_id= '12032';
select agency_id
from m.agency
where agency_id_id= '2991'
  and valid_now=2978;
select agency_id
from m.agency
where agency_id_id= '459'
  and valid_now=40;
select user_id
from m.agency
where valid_now=1664
  and agency_id_id= '2378';
select COUNT(*)
from dv.notes_message
where user_id='7396'
  and agency_id_id= '7396'
  and notice_id= '7396'
  and route_id= '7396';
select agency_id
from m.agency
where agency_id_id= '1268'
  and valid_now=8811;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5329'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12950'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4969'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6460'
  and valid_now=10003;
select COUNT(*)
from dv.notes_message
where user_id='7080'
  and agency_id_id= '7080'
  and notice_id= '7080'
  and route_id= '7080';
select agency_id
from m.agency
where agency_id_id= '2085'
  and valid_now=8913;
select COUNT(*)
from dv.notes_message
where user_id='17148'
  and agency_id_id= '17148'
  and notice_id= '17148'
  and route_id= '17148';
select COUNT(*)
from dv.notes_message
where user_id='13911'
  and agency_id_id= '13911'
  and notice_id= '13911'
  and route_id= '13911';
select COUNT(*)
from dv.notes_message
where user_id='3325'
  and agency_id_id= '3325'
  and notice_id= '3325'
  and route_id= '3325';
select user_id
from m.agency
where valid_now=3989
  and agency_id_id= '17664';
select COUNT(*)
from dv.notes_message
where user_id='1169'
  and agency_id_id= '1169'
  and notice_id= '1169'
  and route_id= '1169';
select COUNT(*)
from dv.notes_message
where user_id='7046'
  and agency_id_id= '7046'
  and notice_id= '7046'
  and route_id= '7046';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13762'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7077'
  and valid_now=8911;
select agency_id
from m.agency
where agency_id_id= '3661'
  and valid_now=17407;
select user_id
from m.agency
where valid_now=16841
  and agency_id_id= '10877';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8861'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2936
  and agency_id_id= '17322';
select COUNT(*)
from dv.notes_message
where user_id='6563'
  and agency_id_id= '6563'
  and notice_id= '6563'
  and route_id= '6563';
select user_id
from m.agency
where valid_now=11847
  and agency_id_id= '9537';
select COUNT(*)
from dv.notes_message
where user_id='16706'
  and agency_id_id= '16706'
  and notice_id= '16706'
  and route_id= '16706';
select agency_id
from m.agency
where agency_id_id= '18359'
  and valid_now=13061;
select agency_id
from m.agency
where agency_id_id= '13649'
  and valid_now=51;
select agency_id
from m.agency
where agency_id_id= '2365'
  and valid_now=18552;
select user_id
from m.agency
where valid_now=735
  and agency_id_id= '10696';
select user_id
from m.agency
where valid_now=18146
  and agency_id_id= '3134';
select COUNT(*)
from dv.notes_message
where user_id='1284'
  and agency_id_id= '1284'
  and notice_id= '1284'
  and route_id= '1284';
select user_id
from m.agency
where valid_now=14695
  and agency_id_id= '6444';
select COUNT(*)
from dv.notes_message
where user_id='6342'
  and agency_id_id= '6342'
  and notice_id= '6342'
  and route_id= '6342';
select user_id
from m.agency
where valid_now=14488
  and agency_id_id= '13889';
select COUNT(*)
from dv.notes_message
where user_id='390'
  and agency_id_id= '390'
  and notice_id= '390'
  and route_id= '390';
select agency_id
from m.agency
where agency_id_id= '9977'
  and valid_now=15817;
select user_id
from m.agency
where valid_now=18184
  and agency_id_id= '11852';
select agency_id
from m.agency
where agency_id_id= '13975'
  and valid_now=14487;
select user_id
from m.agency
where valid_now=10320
  and agency_id_id= '3103';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5809'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16743'
  and agency_id_id= '16743'
  and notice_id= '16743'
  and route_id= '16743';
select agency_id
from m.agency
where agency_id_id= '18550'
  and valid_now=11257;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13694'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18273';
select a.agency_timezone
from m.agency a
where a.agency_id = '9825';
select COUNT(*)
from dv.notes_message
where user_id='2278'
  and agency_id_id= '2278'
  and notice_id= '2278'
  and route_id= '2278';
select a.agency_timezone
from m.agency a
where a.agency_id = '14650';
select COUNT(*)
from dv.notes_message
where user_id='730'
  and agency_id_id= '730'
  and notice_id= '730'
  and route_id= '730';
select a.agency_timezone
from m.agency a
where a.agency_id = '4235';
select a.agency_timezone
from m.agency a
where a.agency_id = '671';
select COUNT(*)
from dv.notes_message
where user_id='12479'
  and agency_id_id= '12479'
  and notice_id= '12479'
  and route_id= '12479';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19288'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10093'
  and valid_now=3386;
select a.agency_timezone
from m.agency a
where a.agency_id = '13135';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19863'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8841'
  and valid_now=13047;
select COUNT(*)
from dv.notes_message
where user_id='3847'
  and agency_id_id= '3847'
  and notice_id= '3847'
  and route_id= '3847';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12649'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1112'
  and valid_now=7149;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6417'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10060'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6104
  and agency_id_id= '4936';
select COUNT(*)
from dv.notes_message
where user_id='12891'
  and agency_id_id= '12891'
  and notice_id= '12891'
  and route_id= '12891';
select agency_id
from m.agency
where agency_id_id= '2844'
  and valid_now=15340;
select COUNT(*)
from dv.notes_message
where user_id='11788'
  and agency_id_id= '11788'
  and notice_id= '11788'
  and route_id= '11788';
select COUNT(*)
from dv.notes_message
where user_id='4021'
  and agency_id_id= '4021'
  and notice_id= '4021'
  and route_id= '4021';
select user_id
from m.agency
where valid_now=7907
  and agency_id_id= '331';
select COUNT(*)
from dv.notes_message
where user_id='6562'
  and agency_id_id= '6562'
  and notice_id= '6562'
  and route_id= '6562';
select agency_id
from m.agency
where agency_id_id= '8823'
  and valid_now=15470;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15947'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19157'
  and agency_id_id= '19157'
  and notice_id= '19157'
  and route_id= '19157';
select agency_id
from m.agency
where agency_id_id= '7259'
  and valid_now=12512;
select agency_id
from m.agency
where agency_id_id= '11640'
  and valid_now=17287;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16659'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16472'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='10435'
  and agency_id_id= '10435'
  and notice_id= '10435'
  and route_id= '10435';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13780'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10453
  and agency_id_id= '1887';
select user_id
from m.agency
where valid_now=13728
  and agency_id_id= '2544';
select agency_id
from m.agency
where agency_id_id= '15771'
  and valid_now=13131;
select COUNT(*)
from dv.notes_message
where user_id='15352'
  and agency_id_id= '15352'
  and notice_id= '15352'
  and route_id= '15352';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10554'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12584'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2988
  and agency_id_id= '6532';
select agency_id
from m.agency
where agency_id_id= '4986'
  and valid_now=12737;
select user_id
from m.agency
where valid_now=15289
  and agency_id_id= '12812';
select user_id
from m.agency
where valid_now=6173
  and agency_id_id= '8755';
select COUNT(*)
from dv.notes_message
where user_id='14999'
  and agency_id_id= '14999'
  and notice_id= '14999'
  and route_id= '14999';
select agency_id
from m.agency
where agency_id_id= '2272'
  and valid_now=1950;
select agency_id
from m.agency
where agency_id_id= '17899'
  and valid_now=13108;
select agency_id
from m.agency
where agency_id_id= '8403'
  and valid_now=5135;
select user_id
from m.agency
where valid_now=19707
  and agency_id_id= '6077';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13424'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13623'
  and valid_now=13380;
select agency_id
from m.agency
where agency_id_id= '8320'
  and valid_now=165;
select user_id
from m.agency
where valid_now=5575
  and agency_id_id= '13752';
select user_id
from m.agency
where valid_now=17832
  and agency_id_id= '16127';
select agency_id
from m.agency
where agency_id_id= '1219'
  and valid_now=9165;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5670'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12926'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17555'
  and valid_now=19997;
select agency_id
from m.agency
where agency_id_id= '10157'
  and valid_now=1176;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2309'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19839'
  and valid_now=16704;
select user_id
from m.agency
where valid_now=17979
  and agency_id_id= '3135';
select user_id
from m.agency
where valid_now=4401
  and agency_id_id= '16597';
select agency_id
from m.agency
where agency_id_id= '12428'
  and valid_now=13338;
select user_id
from m.agency
where valid_now=2648
  and agency_id_id= '14887';
select COUNT(*)
from dv.notes_message
where user_id='11488'
  and agency_id_id= '11488'
  and notice_id= '11488'
  and route_id= '11488';
select COUNT(*)
from dv.notes_message
where user_id='1311'
  and agency_id_id= '1311'
  and notice_id= '1311'
  and route_id= '1311';
select COUNT(*)
from dv.notes_message
where user_id='16093'
  and agency_id_id= '16093'
  and notice_id= '16093'
  and route_id= '16093';
select agency_id
from m.agency
where agency_id_id= '116'
  and valid_now=4874;
select agency_id
from m.agency
where agency_id_id= '6073'
  and valid_now=11304;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16625'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14700'
  and valid_now=5304;
select COUNT(*)
from dv.notes_message
where user_id='540'
  and agency_id_id= '540'
  and notice_id= '540'
  and route_id= '540';
select COUNT(*)
from dv.notes_message
where user_id='14461'
  and agency_id_id= '14461'
  and notice_id= '14461'
  and route_id= '14461';
select COUNT(*)
from dv.notes_message
where user_id='4507'
  and agency_id_id= '4507'
  and notice_id= '4507'
  and route_id= '4507';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4748'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11866'
  and valid_now=6640;
select COUNT(*)
from dv.notes_message
where user_id='19495'
  and agency_id_id= '19495'
  and notice_id= '19495'
  and route_id= '19495';
select agency_id
from m.agency
where agency_id_id= '19925'
  and valid_now=18010;
select COUNT(*)
from dv.notes_message
where user_id='19902'
  and agency_id_id= '19902'
  and notice_id= '19902'
  and route_id= '19902';
select agency_id
from m.agency
where agency_id_id= '7899'
  and valid_now=18150;
select agency_id
from m.agency
where agency_id_id= '13655'
  and valid_now=13164;
select agency_id
from m.agency
where agency_id_id= '14021'
  and valid_now=6247;
select agency_id
from m.agency
where agency_id_id= '8771'
  and valid_now=11433;
select user_id
from m.agency
where valid_now=1506
  and agency_id_id= '14739';
select COUNT(*)
from dv.notes_message
where user_id='8079'
  and agency_id_id= '8079'
  and notice_id= '8079'
  and route_id= '8079';
select agency_id
from m.agency
where agency_id_id= '453'
  and valid_now=1723;
select user_id
from m.agency
where valid_now=9278
  and agency_id_id= '15267';
select user_id
from m.agency
where valid_now=19046
  and agency_id_id= '6404';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16722'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17230'
  and valid_now=18407;
select agency_id
from m.agency
where agency_id_id= '14077'
  and valid_now=11179;
select COUNT(*)
from dv.notes_message
where user_id='10361'
  and agency_id_id= '10361'
  and notice_id= '10361'
  and route_id= '10361';
select agency_id
from m.agency
where agency_id_id= '12640'
  and valid_now=8230;
select agency_id
from m.agency
where agency_id_id= '14061'
  and valid_now=12780;
select user_id
from m.agency
where valid_now=1580
  and agency_id_id= '7111';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '204'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '828'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2354'
  and valid_now=12862;
select user_id
from m.agency
where valid_now=11574
  and agency_id_id= '6481';
select agency_id
from m.agency
where agency_id_id= '2168'
  and valid_now=1810;
select agency_id
from m.agency
where agency_id_id= '7676'
  and valid_now=12816;
select a.agency_timezone
from m.agency a
where a.agency_id = '18837';
select user_id
from m.agency
where valid_now=305
  and agency_id_id= '5089';
select agency_id
from m.agency
where agency_id_id= '18818'
  and valid_now=963;
select a.agency_timezone
from m.agency a
where a.agency_id = '12743';
select user_id
from m.agency
where valid_now=2625
  and agency_id_id= '16848';
select COUNT(*)
from dv.notes_message
where user_id='6039'
  and agency_id_id= '6039'
  and notice_id= '6039'
  and route_id= '6039';
select agency_id
from m.agency
where agency_id_id= '1736'
  and valid_now=8516;
select user_id
from m.agency
where valid_now=12026
  and agency_id_id= '3775';
select COUNT(*)
from dv.notes_message
where user_id='313'
  and agency_id_id= '313'
  and notice_id= '313'
  and route_id= '313';
select a.agency_timezone
from m.agency a
where a.agency_id = '1554';
select agency_id
from m.agency
where agency_id_id= '15203'
  and valid_now=13756;
select agency_id
from m.agency
where agency_id_id= '4879'
  and valid_now=16059;
select COUNT(*)
from dv.notes_message
where user_id='3999'
  and agency_id_id= '3999'
  and notice_id= '3999'
  and route_id= '3999';
select COUNT(*)
from dv.notes_message
where user_id='1356'
  and agency_id_id= '1356'
  and notice_id= '1356'
  and route_id= '1356';
select agency_id
from m.agency
where agency_id_id= '10047'
  and valid_now=8309;
select COUNT(*)
from dv.notes_message
where user_id='18797'
  and agency_id_id= '18797'
  and notice_id= '18797'
  and route_id= '18797';
select COUNT(*)
from dv.notes_message
where user_id='16506'
  and agency_id_id= '16506'
  and notice_id= '16506'
  and route_id= '16506';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12413'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10421
  and agency_id_id= '16675';
select COUNT(*)
from dv.notes_message
where user_id='6612'
  and agency_id_id= '6612'
  and notice_id= '6612'
  and route_id= '6612';
select COUNT(*)
from dv.notes_message
where user_id='326'
  and agency_id_id= '326'
  and notice_id= '326'
  and route_id= '326';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15781'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15747
  and agency_id_id= '5198';
select user_id
from m.agency
where valid_now=1737
  and agency_id_id= '16775';
select COUNT(*)
from dv.notes_message
where user_id='13556'
  and agency_id_id= '13556'
  and notice_id= '13556'
  and route_id= '13556';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15277'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7694'
  and agency_id_id= '7694'
  and notice_id= '7694'
  and route_id= '7694';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12728'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12713
  and agency_id_id= '16001';
select user_id
from m.agency
where valid_now=3411
  and agency_id_id= '5430';
select COUNT(*)
from dv.notes_message
where user_id='17407'
  and agency_id_id= '17407'
  and notice_id= '17407'
  and route_id= '17407';
select COUNT(*)
from dv.notes_message
where user_id='15039'
  and agency_id_id= '15039'
  and notice_id= '15039'
  and route_id= '15039';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3578'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '520'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19479'
  and valid_now=8661;
select user_id
from m.agency
where valid_now=15131
  and agency_id_id= '18286';
select user_id
from m.agency
where valid_now=535
  and agency_id_id= '9527';
select COUNT(*)
from dv.notes_message
where user_id='16710'
  and agency_id_id= '16710'
  and notice_id= '16710'
  and route_id= '16710';
select agency_id
from m.agency
where agency_id_id= '19027'
  and valid_now=1075;
select agency_id
from m.agency
where agency_id_id= '9842'
  and valid_now=10424;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18155'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13605
  and agency_id_id= '16549';
select COUNT(*)
from dv.notes_message
where user_id='7478'
  and agency_id_id= '7478'
  and notice_id= '7478'
  and route_id= '7478';
select user_id
from m.agency
where valid_now=16257
  and agency_id_id= '17101';
select COUNT(*)
from dv.notes_message
where user_id='69'
  and agency_id_id= '69'
  and notice_id= '69'
  and route_id= '69';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1042'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4710
  and agency_id_id= '17703';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15263'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7453'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3023'
  and agency_id_id= '3023'
  and notice_id= '3023'
  and route_id= '3023';
select COUNT(*)
from dv.notes_message
where user_id='15290'
  and agency_id_id= '15290'
  and notice_id= '15290'
  and route_id= '15290';
select COUNT(*)
from dv.notes_message
where user_id='19372'
  and agency_id_id= '19372'
  and notice_id= '19372'
  and route_id= '19372';
select COUNT(*)
from dv.notes_message
where user_id='12955'
  and agency_id_id= '12955'
  and notice_id= '12955'
  and route_id= '12955';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12217'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3247'
  and agency_id_id= '3247'
  and notice_id= '3247'
  and route_id= '3247';
select agency_id
from m.agency
where agency_id_id= '11755'
  and valid_now=16210;
select COUNT(*)
from dv.notes_message
where user_id='17738'
  and agency_id_id= '17738'
  and notice_id= '17738'
  and route_id= '17738';
select user_id
from m.agency
where valid_now=10422
  and agency_id_id= '5212';
select agency_id
from m.agency
where agency_id_id= '2743'
  and valid_now=15451;
select COUNT(*)
from dv.notes_message
where user_id='18753'
  and agency_id_id= '18753'
  and notice_id= '18753'
  and route_id= '18753';
select COUNT(*)
from dv.notes_message
where user_id='834'
  and agency_id_id= '834'
  and notice_id= '834'
  and route_id= '834';
select COUNT(*)
from dv.notes_message
where user_id='9917'
  and agency_id_id= '9917'
  and notice_id= '9917'
  and route_id= '9917';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12882'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15695'
  and valid_now=1258;
select agency_id
from m.agency
where agency_id_id= '12077'
  and valid_now=14262;
select agency_id
from m.agency
where agency_id_id= '19333'
  and valid_now=7506;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9506'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12419'
  and agency_id_id= '12419'
  and notice_id= '12419'
  and route_id= '12419';
select agency_id
from m.agency
where agency_id_id= '13843'
  and valid_now=7050;
select COUNT(*)
from dv.notes_message
where user_id='11411'
  and agency_id_id= '11411'
  and notice_id= '11411'
  and route_id= '11411';
select agency_id
from m.agency
where agency_id_id= '12058'
  and valid_now=18723;
select COUNT(*)
from dv.notes_message
where user_id='19423'
  and agency_id_id= '19423'
  and notice_id= '19423'
  and route_id= '19423';
select COUNT(*)
from dv.notes_message
where user_id='16030'
  and agency_id_id= '16030'
  and notice_id= '16030'
  and route_id= '16030';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10991'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1748'
  and valid_now=13946;
select user_id
from m.agency
where valid_now=19951
  and agency_id_id= '3805';
select user_id
from m.agency
where valid_now=9289
  and agency_id_id= '13466';
select COUNT(*)
from dv.notes_message
where user_id='8061'
  and agency_id_id= '8061'
  and notice_id= '8061'
  and route_id= '8061';
select COUNT(*)
from dv.notes_message
where user_id='18384'
  and agency_id_id= '18384'
  and notice_id= '18384'
  and route_id= '18384';
select COUNT(*)
from dv.notes_message
where user_id='8032'
  and agency_id_id= '8032'
  and notice_id= '8032'
  and route_id= '8032';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6667'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19479'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19988'
  and agency_id_id= '19988'
  and notice_id= '19988'
  and route_id= '19988';
select COUNT(*)
from dv.notes_message
where user_id='13273'
  and agency_id_id= '13273'
  and notice_id= '13273'
  and route_id= '13273';
select agency_id
from m.agency
where agency_id_id= '9095'
  and valid_now=6977;
select agency_id
from m.agency
where agency_id_id= '16652'
  and valid_now=18081;
select user_id
from m.agency
where valid_now=15252
  and agency_id_id= '16351';
select user_id
from m.agency
where valid_now=11535
  and agency_id_id= '8149';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2425'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '366'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3223'
  and valid_now=19219;
select user_id
from m.agency
where valid_now=13800
  and agency_id_id= '13270';
select user_id
from m.agency
where valid_now=8640
  and agency_id_id= '3544';
select user_id
from m.agency
where valid_now=5769
  and agency_id_id= '9295';
select COUNT(*)
from dv.notes_message
where user_id='2911'
  and agency_id_id= '2911'
  and notice_id= '2911'
  and route_id= '2911';
select COUNT(*)
from dv.notes_message
where user_id='17310'
  and agency_id_id= '17310'
  and notice_id= '17310'
  and route_id= '17310';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6846'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8962
  and agency_id_id= '14039';
select COUNT(*)
from dv.notes_message
where user_id='12297'
  and agency_id_id= '12297'
  and notice_id= '12297'
  and route_id= '12297';
select user_id
from m.agency
where valid_now=8201
  and agency_id_id= '1436';
select user_id
from m.agency
where valid_now=13672
  and agency_id_id= '13129';
select user_id
from m.agency
where valid_now=17518
  and agency_id_id= '17248';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4461'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17490'
  and agency_id_id= '17490'
  and notice_id= '17490'
  and route_id= '17490';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5845'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8831
  and agency_id_id= '17327';
select COUNT(*)
from dv.notes_message
where user_id='2932'
  and agency_id_id= '2932'
  and notice_id= '2932'
  and route_id= '2932';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14251'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17995'
  and valid_now=13030;
select agency_id
from m.agency
where agency_id_id= '8028'
  and valid_now=18777;
select user_id
from m.agency
where valid_now=19841
  and agency_id_id= '14781';
select user_id
from m.agency
where valid_now=13365
  and agency_id_id= '1503';
select COUNT(*)
from dv.notes_message
where user_id='8014'
  and agency_id_id= '8014'
  and notice_id= '8014'
  and route_id= '8014';
select COUNT(*)
from dv.notes_message
where user_id='14062'
  and agency_id_id= '14062'
  and notice_id= '14062'
  and route_id= '14062';
select COUNT(*)
from dv.notes_message
where user_id='2094'
  and agency_id_id= '2094'
  and notice_id= '2094'
  and route_id= '2094';
select user_id
from m.agency
where valid_now=13540
  and agency_id_id= '15900';
select COUNT(*)
from dv.notes_message
where user_id='80'
  and agency_id_id= '80'
  and notice_id= '80'
  and route_id= '80';
select COUNT(*)
from dv.notes_message
where user_id='15494'
  and agency_id_id= '15494'
  and notice_id= '15494'
  and route_id= '15494';
select agency_id
from m.agency
where agency_id_id= '4016'
  and valid_now=12410;
select user_id
from m.agency
where valid_now=3681
  and agency_id_id= '13254';
select user_id
from m.agency
where valid_now=8791
  and agency_id_id= '4100';
select agency_id
from m.agency
where agency_id_id= '19837'
  and valid_now=11346;
select agency_id
from m.agency
where agency_id_id= '14762'
  and valid_now=1039;
select agency_id
from m.agency
where agency_id_id= '16161'
  and valid_now=14726;
select COUNT(*)
from dv.notes_message
where user_id='240'
  and agency_id_id= '240'
  and notice_id= '240'
  and route_id= '240';
select user_id
from m.agency
where valid_now=2994
  and agency_id_id= '9576';
select user_id
from m.agency
where valid_now=11289
  and agency_id_id= '15806';
select user_id
from m.agency
where valid_now=18769
  and agency_id_id= '14544';
select COUNT(*)
from dv.notes_message
where user_id='6579'
  and agency_id_id= '6579'
  and notice_id= '6579'
  and route_id= '6579';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15727'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3514
  and agency_id_id= '5604';
select user_id
from m.agency
where valid_now=14686
  and agency_id_id= '17073';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14643'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19444'
  and agency_id_id= '19444'
  and notice_id= '19444'
  and route_id= '19444';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16766'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11756'
  and valid_now=5;
select COUNT(*)
from dv.notes_message
where user_id='8048'
  and agency_id_id= '8048'
  and notice_id= '8048'
  and route_id= '8048';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2750'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14971'
  and valid_now=6479;
select agency_id
from m.agency
where agency_id_id= '1730'
  and valid_now=4988;
select user_id
from m.agency
where valid_now=8991
  and agency_id_id= '16412';
select user_id
from m.agency
where valid_now=3229
  and agency_id_id= '7671';
select agency_id
from m.agency
where agency_id_id= '14897'
  and valid_now=8128;
select COUNT(*)
from dv.notes_message
where user_id='18587'
  and agency_id_id= '18587'
  and notice_id= '18587'
  and route_id= '18587';
select user_id
from m.agency
where valid_now=5705
  and agency_id_id= '10855';
select user_id
from m.agency
where valid_now=11197
  and agency_id_id= '13223';
select COUNT(*)
from dv.notes_message
where user_id='1277'
  and agency_id_id= '1277'
  and notice_id= '1277'
  and route_id= '1277';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15529'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5579'
  and valid_now=17814;
select user_id
from m.agency
where valid_now=14449
  and agency_id_id= '8053';
select user_id
from m.agency
where valid_now=16442
  and agency_id_id= '15297';
select COUNT(*)
from dv.notes_message
where user_id='14982'
  and agency_id_id= '14982'
  and notice_id= '14982'
  and route_id= '14982';
select COUNT(*)
from dv.notes_message
where user_id='4768'
  and agency_id_id= '4768'
  and notice_id= '4768'
  and route_id= '4768';
select user_id
from m.agency
where valid_now=19219
  and agency_id_id= '6071';
select user_id
from m.agency
where valid_now=9892
  and agency_id_id= '8838';
select agency_id
from m.agency
where agency_id_id= '6726'
  and valid_now=8490;
select user_id
from m.agency
where valid_now=11856
  and agency_id_id= '8067';
select COUNT(*)
from dv.notes_message
where user_id='9740'
  and agency_id_id= '9740'
  and notice_id= '9740'
  and route_id= '9740';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8378'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5001'
  and agency_id_id= '5001'
  and notice_id= '5001'
  and route_id= '5001';
select COUNT(*)
from dv.notes_message
where user_id='20'
  and agency_id_id= '20'
  and notice_id= '20'
  and route_id= '20';
select agency_id
from m.agency
where agency_id_id= '509'
  and valid_now=7546;
select agency_id
from m.agency
where agency_id_id= '16319'
  and valid_now=16398;
select COUNT(*)
from dv.notes_message
where user_id='15108'
  and agency_id_id= '15108'
  and notice_id= '15108'
  and route_id= '15108';
select agency_id
from m.agency
where agency_id_id= '3804'
  and valid_now=11178;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15091'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14575'
  and valid_now=15919;
select a.agency_timezone
from m.agency a
where a.agency_id = '5953';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10060'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16959'
  and agency_id_id= '16959'
  and notice_id= '16959'
  and route_id= '16959';
select agency_id
from m.agency
where agency_id_id= '2846'
  and valid_now=10753;
select COUNT(*)
from dv.notes_message
where user_id='117'
  and agency_id_id= '117'
  and notice_id= '117'
  and route_id= '117';
select COUNT(*)
from dv.notes_message
where user_id='1260'
  and agency_id_id= '1260'
  and notice_id= '1260'
  and route_id= '1260';
select COUNT(*)
from dv.notes_message
where user_id='13841'
  and agency_id_id= '13841'
  and notice_id= '13841'
  and route_id= '13841';
select COUNT(*)
from dv.notes_message
where user_id='17938'
  and agency_id_id= '17938'
  and notice_id= '17938'
  and route_id= '17938';
select a.agency_timezone
from m.agency a
where a.agency_id = '10651';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16239'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5126'
  and valid_now=11613;
select COUNT(*)
from dv.notes_message
where user_id='1810'
  and agency_id_id= '1810'
  and notice_id= '1810'
  and route_id= '1810';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_5777'
  and t.trip_id = 16918
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11592'
  and agency_id_id= '11592'
  and notice_id= '11592'
  and route_id= '11592';
select COUNT(*)
from dv.notes_message
where user_id='3344'
  and agency_id_id= '3344'
  and notice_id= '3344'
  and route_id= '3344';
select user_id
from m.agency
where valid_now=1963
  and agency_id_id= '5095';
select COUNT(*)
from dv.notes_message
where user_id='12250'
  and agency_id_id= '12250'
  and notice_id= '12250'
  and route_id= '12250';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8432'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2597'
  and agency_id_id= '2597'
  and notice_id= '2597'
  and route_id= '2597';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15232'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16159
  and agency_id_id= '11210';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19863'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3453'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6690'
  and agency_id_id= '6690'
  and notice_id= '6690'
  and route_id= '6690';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12046'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6722'
  and valid_now=1440;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8611'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3716
  and agency_id_id= '13081';
select user_id
from m.agency
where valid_now=10867
  and agency_id_id= '3458';
select COUNT(*)
from dv.notes_message
where user_id='3416'
  and agency_id_id= '3416'
  and notice_id= '3416'
  and route_id= '3416';
select user_id
from m.agency
where valid_now=14135
  and agency_id_id= '1621';
select COUNT(*)
from dv.notes_message
where user_id='6422'
  and agency_id_id= '6422'
  and notice_id= '6422'
  and route_id= '6422';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15481'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7320
  and agency_id_id= '14508';
select agency_id
from m.agency
where agency_id_id= '11751'
  and valid_now=11883;
select COUNT(*)
from dv.notes_message
where user_id='3889'
  and agency_id_id= '3889'
  and notice_id= '3889'
  and route_id= '3889';
select COUNT(*)
from dv.notes_message
where user_id='8362'
  and agency_id_id= '8362'
  and notice_id= '8362'
  and route_id= '8362';
select agency_id
from m.agency
where agency_id_id= '127'
  and valid_now=3099;
select user_id
from m.agency
where valid_now=7002
  and agency_id_id= '8177';
select user_id
from m.agency
where valid_now=16133
  and agency_id_id= '16321';
select COUNT(*)
from dv.notes_message
where user_id='12406'
  and agency_id_id= '12406'
  and notice_id= '12406'
  and route_id= '12406';
select agency_id
from m.agency
where agency_id_id= '16026'
  and valid_now=12867;
select agency_id
from m.agency
where agency_id_id= '12188'
  and valid_now=924;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1778'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13998
  and agency_id_id= '9581';
select user_id
from m.agency
where valid_now=10872
  and agency_id_id= '10608';
select COUNT(*)
from dv.notes_message
where user_id='181'
  and agency_id_id= '181'
  and notice_id= '181'
  and route_id= '181';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1090'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5994'
  and valid_now=6892;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19319'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=423
  and agency_id_id= '13040';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12533'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11499'
  and valid_now=17730;
select agency_id
from m.agency
where agency_id_id= '7985'
  and valid_now=8536;
select user_id
from m.agency
where valid_now=3606
  and agency_id_id= '6589';
select COUNT(*)
from dv.notes_message
where user_id='10471'
  and agency_id_id= '10471'
  and notice_id= '10471'
  and route_id= '10471';
select COUNT(*)
from dv.notes_message
where user_id='8215'
  and agency_id_id= '8215'
  and notice_id= '8215'
  and route_id= '8215';
select COUNT(*)
from dv.notes_message
where user_id='23'
  and agency_id_id= '23'
  and notice_id= '23'
  and route_id= '23';
select user_id
from m.agency
where valid_now=13935
  and agency_id_id= '19542';
select COUNT(*)
from dv.notes_message
where user_id='11079'
  and agency_id_id= '11079'
  and notice_id= '11079'
  and route_id= '11079';
select user_id
from m.agency
where valid_now=2977
  and agency_id_id= '5141';
select user_id
from m.agency
where valid_now=9350
  and agency_id_id= '17376';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6316'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12069'
  and valid_now=1657;
select agency_id
from m.agency
where agency_id_id= '4176'
  and valid_now=10425;
select user_id
from m.agency
where valid_now=8460
  and agency_id_id= '17948';
select COUNT(*)
from dv.notes_message
where user_id='7169'
  and agency_id_id= '7169'
  and notice_id= '7169'
  and route_id= '7169';
select agency_id
from m.agency
where agency_id_id= '10549'
  and valid_now=13558;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2016'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16040'
  and valid_now=16275;
select COUNT(*)
from dv.notes_message
where user_id='11347'
  and agency_id_id= '11347'
  and notice_id= '11347'
  and route_id= '11347';
select agency_id
from m.agency
where agency_id_id= '8118'
  and valid_now=10741;
select agency_id
from m.agency
where agency_id_id= '11366'
  and valid_now=6071;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19216'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4043'
  and valid_now=16925;
select COUNT(*)
from dv.notes_message
where user_id='16995'
  and agency_id_id= '16995'
  and notice_id= '16995'
  and route_id= '16995';
select user_id
from m.agency
where valid_now=15756
  and agency_id_id= '15792';
select agency_id
from m.agency
where agency_id_id= '5599'
  and valid_now=18544;
select user_id
from m.agency
where valid_now=6599
  and agency_id_id= '13258';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11005'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16644
  and agency_id_id= '15713';
select COUNT(*)
from dv.notes_message
where user_id='4949'
  and agency_id_id= '4949'
  and notice_id= '4949'
  and route_id= '4949';
select user_id
from m.agency
where valid_now=1301
  and agency_id_id= '370';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1579'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6911'
  and valid_now=4741;
select user_id
from m.agency
where valid_now=17390
  and agency_id_id= '7799';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18234'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13651'
  and valid_now=16457;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16848'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19932'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9292'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5304
  and agency_id_id= '15905';
select user_id
from m.agency
where valid_now=14150
  and agency_id_id= '19527';
select COUNT(*)
from dv.notes_message
where user_id='9414'
  and agency_id_id= '9414'
  and notice_id= '9414'
  and route_id= '9414';
select agency_id
from m.agency
where agency_id_id= '5121'
  and valid_now=19854;
select agency_id
from m.agency
where agency_id_id= '11832'
  and valid_now=307;
select agency_id
from m.agency
where agency_id_id= '11169'
  and valid_now=7550;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14974'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5192'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18880
  and agency_id_id= '6108';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_10051'
  and t.trip_id = 799
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16220'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14776
  and agency_id_id= '15481';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_15450'
  and t.trip_id = 11219
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10316'
  and valid_now=7750;
select a.agency_timezone
from m.agency a
where a.agency_id = '4833';
select user_id
from m.agency
where valid_now=15681
  and agency_id_id= '1405';
select COUNT(*)
from dv.notes_message
where user_id='4556'
  and agency_id_id= '4556'
  and notice_id= '4556'
  and route_id= '4556';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '969'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15894'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16807'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19211'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2372'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9290'
  and agency_id_id= '9290'
  and notice_id= '9290'
  and route_id= '9290';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17670'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6941'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17872'
  and valid_now=16538;
select user_id
from m.agency
where valid_now=18071
  and agency_id_id= '1402';
select user_id
from m.agency
where valid_now=4333
  and agency_id_id= '11945';
select user_id
from m.agency
where valid_now=1645
  and agency_id_id= '16890';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19187'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11826
  and agency_id_id= '13315';
select agency_id
from m.agency
where agency_id_id= '8043'
  and valid_now=16253;
select COUNT(*)
from dv.notes_message
where user_id='15900'
  and agency_id_id= '15900'
  and notice_id= '15900'
  and route_id= '15900';
select agency_id
from m.agency
where agency_id_id= '10945'
  and valid_now=14748;
select COUNT(*)
from dv.notes_message
where user_id='16567'
  and agency_id_id= '16567'
  and notice_id= '16567'
  and route_id= '16567';
select agency_id
from m.agency
where agency_id_id= '16930'
  and valid_now=5914;
select COUNT(*)
from dv.notes_message
where user_id='3521'
  and agency_id_id= '3521'
  and notice_id= '3521'
  and route_id= '3521';
select user_id
from m.agency
where valid_now=17771
  and agency_id_id= '7505';
select a.agency_timezone
from m.agency a
where a.agency_id = '8908';
select user_id
from m.agency
where valid_now=10907
  and agency_id_id= '18667';
select user_id
from m.agency
where valid_now=15187
  and agency_id_id= '10468';
select COUNT(*)
from dv.notes_message
where user_id='9712'
  and agency_id_id= '9712'
  and notice_id= '9712'
  and route_id= '9712';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12839'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '2976';
select COUNT(*)
from dv.notes_message
where user_id='19109'
  and agency_id_id= '19109'
  and notice_id= '19109'
  and route_id= '19109';
select a.agency_timezone
from m.agency a
where a.agency_id = '6174';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1328'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8109'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16809'
  and agency_id_id= '16809'
  and notice_id= '16809'
  and route_id= '16809';
select a.agency_timezone
from m.agency a
where a.agency_id = '17868';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16803'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18063';
select COUNT(*)
from dv.notes_message
where user_id='6620'
  and agency_id_id= '6620'
  and notice_id= '6620'
  and route_id= '6620';
select COUNT(*)
from dv.notes_message
where user_id='2238'
  and agency_id_id= '2238'
  and notice_id= '2238'
  and route_id= '2238';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3504'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11615';
select user_id
from m.agency
where valid_now=5799
  and agency_id_id= '15087';
select COUNT(*)
from dv.notes_message
where user_id='18576'
  and agency_id_id= '18576'
  and notice_id= '18576'
  and route_id= '18576';
select a.agency_timezone
from m.agency a
where a.agency_id = '7927';
select COUNT(*)
from dv.notes_message
where user_id='16366'
  and agency_id_id= '16366'
  and notice_id= '16366'
  and route_id= '16366';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13639'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18867'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6807'
  and valid_now=3929;
select agency_id
from m.agency
where agency_id_id= '996'
  and valid_now=16696;
select COUNT(*)
from dv.notes_message
where user_id='17870'
  and agency_id_id= '17870'
  and notice_id= '17870'
  and route_id= '17870';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4052'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4101'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3195'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10237'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4548'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9323'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19284'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '818'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19173'
  and valid_now=19011;
select COUNT(*)
from dv.notes_message
where user_id='15776'
  and agency_id_id= '15776'
  and notice_id= '15776'
  and route_id= '15776';
select COUNT(*)
from dv.notes_message
where user_id='7057'
  and agency_id_id= '7057'
  and notice_id= '7057'
  and route_id= '7057';
select agency_id
from m.agency
where agency_id_id= '11441'
  and valid_now=225;
select agency_id
from m.agency
where agency_id_id= '5155'
  and valid_now=3074;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14219'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19563'
  and agency_id_id= '19563'
  and notice_id= '19563'
  and route_id= '19563';
select agency_id
from m.agency
where agency_id_id= '13745'
  and valid_now=3305;
select COUNT(*)
from dv.notes_message
where user_id='16929'
  and agency_id_id= '16929'
  and notice_id= '16929'
  and route_id= '16929';
select user_id
from m.agency
where valid_now=1986
  and agency_id_id= '10535';
select COUNT(*)
from dv.notes_message
where user_id='14091'
  and agency_id_id= '14091'
  and notice_id= '14091'
  and route_id= '14091';
select agency_id
from m.agency
where agency_id_id= '19957'
  and valid_now=1536;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7454'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9507'
  and valid_now=17832;
select agency_id
from m.agency
where agency_id_id= '17480'
  and valid_now=14568;
select agency_id
from m.agency
where agency_id_id= '18389'
  and valid_now=15429;
select agency_id
from m.agency
where agency_id_id= '316'
  and valid_now=10708;
select user_id
from m.agency
where valid_now=8724
  and agency_id_id= '10843';
select COUNT(*)
from dv.notes_message
where user_id='19506'
  and agency_id_id= '19506'
  and notice_id= '19506'
  and route_id= '19506';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7397'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7837'
  and valid_now=8789;
select user_id
from m.agency
where valid_now=7902
  and agency_id_id= '19157';
select COUNT(*)
from dv.notes_message
where user_id='16204'
  and agency_id_id= '16204'
  and notice_id= '16204'
  and route_id= '16204';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15900'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12381'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5451'
  and agency_id_id= '5451'
  and notice_id= '5451'
  and route_id= '5451';
select user_id
from m.agency
where valid_now=3031
  and agency_id_id= '4413';
select agency_id
from m.agency
where agency_id_id= '18511'
  and valid_now=11816;
select user_id
from m.agency
where valid_now=5729
  and agency_id_id= '17553';
select agency_id
from m.agency
where agency_id_id= '13814'
  and valid_now=14044;
select agency_id
from m.agency
where agency_id_id= '11461'
  and valid_now=18247;
select user_id
from m.agency
where valid_now=15833
  and agency_id_id= '7280';
select user_id
from m.agency
where valid_now=15521
  and agency_id_id= '11165';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13105'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5079'
  and valid_now=12938;
select user_id
from m.agency
where valid_now=10918
  and agency_id_id= '15683';
select COUNT(*)
from dv.notes_message
where user_id='17823'
  and agency_id_id= '17823'
  and notice_id= '17823'
  and route_id= '17823';
select COUNT(*)
from dv.notes_message
where user_id='15008'
  and agency_id_id= '15008'
  and notice_id= '15008'
  and route_id= '15008';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3395'
  and agency_id_id= '3395'
  and notice_id= '3395'
  and route_id= '3395';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18513'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1857
  and agency_id_id= '4120';
select user_id
from m.agency
where valid_now=18411
  and agency_id_id= '16212';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4597'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1393'
  and valid_now=14172;
select user_id
from m.agency
where valid_now=11645
  and agency_id_id= '11130';
select COUNT(*)
from dv.notes_message
where user_id='8671'
  and agency_id_id= '8671'
  and notice_id= '8671'
  and route_id= '8671';
select user_id
from m.agency
where valid_now=2907
  and agency_id_id= '5455';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18002'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4840'
  and valid_now=13640;
select user_id
from m.agency
where valid_now=1481
  and agency_id_id= '15262';
select COUNT(*)
from dv.notes_message
where user_id='12906'
  and agency_id_id= '12906'
  and notice_id= '12906'
  and route_id= '12906';
select user_id
from m.agency
where valid_now=15243
  and agency_id_id= '8992';
select user_id
from m.agency
where valid_now=13123
  and agency_id_id= '19507';
select agency_id
from m.agency
where agency_id_id= '11215'
  and valid_now=14535;
select user_id
from m.agency
where valid_now=9188
  and agency_id_id= '6038';
select agency_id
from m.agency
where agency_id_id= '17910'
  and valid_now=5941;
select COUNT(*)
from dv.notes_message
where user_id='9773'
  and agency_id_id= '9773'
  and notice_id= '9773'
  and route_id= '9773';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11474'
  and agency_id_id= '11474'
  and notice_id= '11474'
  and route_id= '11474';
select agency_id
from m.agency
where agency_id_id= '13705'
  and valid_now=12418;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7691'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6216
  and agency_id_id= '12891';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5776'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=921
  and agency_id_id= '11981';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11752'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2183
  and agency_id_id= '13930';
select COUNT(*)
from dv.notes_message
where user_id='12025'
  and agency_id_id= '12025'
  and notice_id= '12025'
  and route_id= '12025';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11409'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9978'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12978'
  and agency_id_id= '12978'
  and notice_id= '12978'
  and route_id= '12978';
select agency_id
from m.agency
where agency_id_id= '12774'
  and valid_now=871;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '900'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19432'
  and agency_id_id= '19432'
  and notice_id= '19432'
  and route_id= '19432';
select agency_id
from m.agency
where agency_id_id= '2869'
  and valid_now=10980;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13437'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9453'
  and agency_id_id= '9453'
  and notice_id= '9453'
  and route_id= '9453';
select COUNT(*)
from dv.notes_message
where user_id='6830'
  and agency_id_id= '6830'
  and notice_id= '6830'
  and route_id= '6830';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8746'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17484'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '602'
  and valid_now=19276;
select COUNT(*)
from dv.notes_message
where user_id='17880'
  and agency_id_id= '17880'
  and notice_id= '17880'
  and route_id= '17880';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6952'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11532'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8641'
  and valid_now=13563;
select agency_id
from m.agency
where agency_id_id= '13433'
  and valid_now=2081;
select COUNT(*)
from dv.notes_message
where user_id='701'
  and agency_id_id= '701'
  and notice_id= '701'
  and route_id= '701';
select agency_id
from m.agency
where agency_id_id= '2834'
  and valid_now=8207;
select agency_id
from m.agency
where agency_id_id= '4456'
  and valid_now=9698;
select COUNT(*)
from dv.notes_message
where user_id='10112'
  and agency_id_id= '10112'
  and notice_id= '10112'
  and route_id= '10112';
select agency_id
from m.agency
where agency_id_id= '18398'
  and valid_now=9482;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6030'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '575'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16361
  and agency_id_id= '16852';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9085'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1643'
  and agency_id_id= '1643'
  and notice_id= '1643'
  and route_id= '1643';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3916'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12702'
  and valid_now=1280;
select user_id
from m.agency
where valid_now=13682
  and agency_id_id= '13869';
select user_id
from m.agency
where valid_now=3733
  and agency_id_id= '3677';
select agency_id
from m.agency
where agency_id_id= '16075'
  and valid_now=2199;
select agency_id
from m.agency
where agency_id_id= '5943'
  and valid_now=18953;
select COUNT(*)
from dv.notes_message
where user_id='9696'
  and agency_id_id= '9696'
  and notice_id= '9696'
  and route_id= '9696';
select agency_id
from m.agency
where agency_id_id= '9339'
  and valid_now=13678;
select user_id
from m.agency
where valid_now=18121
  and agency_id_id= '10335';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7173'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7582'
  and valid_now=2249;
select COUNT(*)
from dv.notes_message
where user_id='18451'
  and agency_id_id= '18451'
  and notice_id= '18451'
  and route_id= '18451';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13349'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16217'
  and valid_now=12537;
select user_id
from m.agency
where valid_now=7783
  and agency_id_id= '17164';
select user_id
from m.agency
where valid_now=5996
  and agency_id_id= '3049';
select user_id
from m.agency
where valid_now=6103
  and agency_id_id= '6885';
select agency_id
from m.agency
where agency_id_id= '624'
  and valid_now=3497;
select agency_id
from m.agency
where agency_id_id= '3930'
  and valid_now=2870;
select agency_id
from m.agency
where agency_id_id= '6505'
  and valid_now=752;
select a.agency_timezone
from m.agency a
where a.agency_id = '4215';
select a.agency_timezone
from m.agency a
where a.agency_id = '12611';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19957'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8513'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11761'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10460'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '7284';
select a.agency_timezone
from m.agency a
where a.agency_id = '19881';
select a.agency_timezone
from m.agency a
where a.agency_id = '4342';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11269'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14184'
  and valid_now=14940;
select agency_id
from m.agency
where agency_id_id= '16884'
  and valid_now=11967;
select a.agency_timezone
from m.agency a
where a.agency_id = '14662';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6381'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13103'
  and agency_id_id= '13103'
  and notice_id= '13103'
  and route_id= '13103';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17374'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18848'
  and agency_id_id= '18848'
  and notice_id= '18848'
  and route_id= '18848';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16479'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9777'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15646'
  and valid_now=6037;
select agency_id
from m.agency
where agency_id_id= '10639'
  and valid_now=8467;
select user_id
from m.agency
where valid_now=2204
  and agency_id_id= '18491';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19213'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5767'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11123
  and agency_id_id= '11793';
select agency_id
from m.agency
where agency_id_id= '11514'
  and valid_now=4551;
select user_id
from m.agency
where valid_now=12484
  and agency_id_id= '9424';
select agency_id
from m.agency
where agency_id_id= '14609'
  and valid_now=8606;
select agency_id
from m.agency
where agency_id_id= '17068'
  and valid_now=17118;
select user_id
from m.agency
where valid_now=5102
  and agency_id_id= '14632';
select user_id
from m.agency
where valid_now=12775
  and agency_id_id= '2485';
select user_id
from m.agency
where valid_now=10350
  and agency_id_id= '19833';
select user_id
from m.agency
where valid_now=4062
  and agency_id_id= '9470';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8070'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3654'
  and valid_now=15807;
select agency_id
from m.agency
where agency_id_id= '9649'
  and valid_now=3445;
select user_id
from m.agency
where valid_now=9035
  and agency_id_id= '3482';
select user_id
from m.agency
where valid_now=13365
  and agency_id_id= '16219';
select agency_id
from m.agency
where agency_id_id= '6814'
  and valid_now=15048;
select COUNT(*)
from dv.notes_message
where user_id='3000'
  and agency_id_id= '3000'
  and notice_id= '3000'
  and route_id= '3000';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3773'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '182'
  and valid_now=10246;
select user_id
from m.agency
where valid_now=12509
  and agency_id_id= '19198';
select COUNT(*)
from dv.notes_message
where user_id='5580'
  and agency_id_id= '5580'
  and notice_id= '5580'
  and route_id= '5580';
select COUNT(*)
from dv.notes_message
where user_id='6780'
  and agency_id_id= '6780'
  and notice_id= '6780'
  and route_id= '6780';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7710'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12804
  and agency_id_id= '19327';
select user_id
from m.agency
where valid_now=3551
  and agency_id_id= '11211';
select COUNT(*)
from dv.notes_message
where user_id='12307'
  and agency_id_id= '12307'
  and notice_id= '12307'
  and route_id= '12307';
select COUNT(*)
from dv.notes_message
where user_id='11617'
  and agency_id_id= '11617'
  and notice_id= '11617'
  and route_id= '11617';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1564'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '699'
  and valid_now=6221;
select user_id
from m.agency
where valid_now=14234
  and agency_id_id= '8032';
select COUNT(*)
from dv.notes_message
where user_id='18062'
  and agency_id_id= '18062'
  and notice_id= '18062'
  and route_id= '18062';
select COUNT(*)
from dv.notes_message
where user_id='9297'
  and agency_id_id= '9297'
  and notice_id= '9297'
  and route_id= '9297';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10455'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11559
  and agency_id_id= '1867';
select COUNT(*)
from dv.notes_message
where user_id='11731'
  and agency_id_id= '11731'
  and notice_id= '11731'
  and route_id= '11731';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12496'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4068'
  and valid_now=1914;
select agency_id
from m.agency
where agency_id_id= '2761'
  and valid_now=1667;
select user_id
from m.agency
where valid_now=7306
  and agency_id_id= '14965';
select agency_id
from m.agency
where agency_id_id= '13282'
  and valid_now=12834;
select user_id
from m.agency
where valid_now=14912
  and agency_id_id= '13612';
select user_id
from m.agency
where valid_now=2331
  and agency_id_id= '1919';
select user_id
from m.agency
where valid_now=5531
  and agency_id_id= '1002';
select agency_id
from m.agency
where agency_id_id= '6852'
  and valid_now=1373;
select user_id
from m.agency
where valid_now=8920
  and agency_id_id= '120';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9571'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1440
  and agency_id_id= '12993';
select agency_id
from m.agency
where agency_id_id= '4855'
  and valid_now=10003;
select agency_id
from m.agency
where agency_id_id= '8787'
  and valid_now=13440;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14018'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12691'
  and valid_now=14289;
select user_id
from m.agency
where valid_now=2132
  and agency_id_id= '17924';
select COUNT(*)
from dv.notes_message
where user_id='19308'
  and agency_id_id= '19308'
  and notice_id= '19308'
  and route_id= '19308';
select COUNT(*)
from dv.notes_message
where user_id='18278'
  and agency_id_id= '18278'
  and notice_id= '18278'
  and route_id= '18278';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17077'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19132
  and agency_id_id= '4265';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4058'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11928'
  and valid_now=13532;
select user_id
from m.agency
where valid_now=13110
  and agency_id_id= '16188';
select user_id
from m.agency
where valid_now=10714
  and agency_id_id= '15529';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4864'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19115'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16035'
  and agency_id_id= '16035'
  and notice_id= '16035'
  and route_id= '16035';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17183'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8364'
  and valid_now=2879;
select user_id
from m.agency
where valid_now=14056
  and agency_id_id= '18005';
select COUNT(*)
from dv.notes_message
where user_id='16611'
  and agency_id_id= '16611'
  and notice_id= '16611'
  and route_id= '16611';
select agency_id
from m.agency
where agency_id_id= '18310'
  and valid_now=9525;
select user_id
from m.agency
where valid_now=14407
  and agency_id_id= '2271';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3801'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18844
  and agency_id_id= '17370';
select user_id
from m.agency
where valid_now=12769
  and agency_id_id= '19301';
select user_id
from m.agency
where valid_now=8297
  and agency_id_id= '18195';
select COUNT(*)
from dv.notes_message
where user_id='17380'
  and agency_id_id= '17380'
  and notice_id= '17380'
  and route_id= '17380';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7369'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18451'
  and valid_now=12681;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2800'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12573'
  and valid_now=19990;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16590'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7644'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8597'
  and agency_id_id= '8597'
  and notice_id= '8597'
  and route_id= '8597';
select COUNT(*)
from dv.notes_message
where user_id='9051'
  and agency_id_id= '9051'
  and notice_id= '9051'
  and route_id= '9051';
select COUNT(*)
from dv.notes_message
where user_id='17319'
  and agency_id_id= '17319'
  and notice_id= '17319'
  and route_id= '17319';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3197'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3644'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2930'
  and valid_now=6295;
select agency_id
from m.agency
where agency_id_id= '1531'
  and valid_now=5562;
select user_id
from m.agency
where valid_now=6565
  and agency_id_id= '763';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3215'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14277'
  and valid_now=6275;
select user_id
from m.agency
where valid_now=17541
  and agency_id_id= '1085';
select user_id
from m.agency
where valid_now=13267
  and agency_id_id= '17368';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13298'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13255'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4033'
  and agency_id_id= '4033'
  and notice_id= '4033'
  and route_id= '4033';
select user_id
from m.agency
where valid_now=11068
  and agency_id_id= '6862';
select COUNT(*)
from dv.notes_message
where user_id='8981'
  and agency_id_id= '8981'
  and notice_id= '8981'
  and route_id= '8981';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14158'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='945'
  and agency_id_id= '945'
  and notice_id= '945'
  and route_id= '945';
select agency_id
from m.agency
where agency_id_id= '18933'
  and valid_now=10411;
select agency_id
from m.agency
where agency_id_id= '14269'
  and valid_now=669;
select agency_id
from m.agency
where agency_id_id= '8963'
  and valid_now=862;
select agency_id
from m.agency
where agency_id_id= '7687'
  and valid_now=8138;
select COUNT(*)
from dv.notes_message
where user_id='17948'
  and agency_id_id= '17948'
  and notice_id= '17948'
  and route_id= '17948';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14930'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16761
  and agency_id_id= '17172';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5318'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6894'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15121'
  and agency_id_id= '15121'
  and notice_id= '15121'
  and route_id= '15121';
select agency_id
from m.agency
where agency_id_id= '6089'
  and valid_now=3370;
select user_id
from m.agency
where valid_now=14801
  and agency_id_id= '1968';
select COUNT(*)
from dv.notes_message
where user_id='12783'
  and agency_id_id= '12783'
  and notice_id= '12783'
  and route_id= '12783';
select agency_id
from m.agency
where agency_id_id= '11564'
  and valid_now=1023;
select agency_id
from m.agency
where agency_id_id= '5360'
  and valid_now=7310;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16687'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5429'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12443
  and agency_id_id= '2827';
select agency_id
from m.agency
where agency_id_id= '17785'
  and valid_now=14971;
select user_id
from m.agency
where valid_now=12761
  and agency_id_id= '13165';
select COUNT(*)
from dv.notes_message
where user_id='13496'
  and agency_id_id= '13496'
  and notice_id= '13496'
  and route_id= '13496';
select user_id
from m.agency
where valid_now=8597
  and agency_id_id= '11370';
select COUNT(*)
from dv.notes_message
where user_id='9500'
  and agency_id_id= '9500'
  and notice_id= '9500'
  and route_id= '9500';
select COUNT(*)
from dv.notes_message
where user_id='3189'
  and agency_id_id= '3189'
  and notice_id= '3189'
  and route_id= '3189';
select COUNT(*)
from dv.notes_message
where user_id='3952'
  and agency_id_id= '3952'
  and notice_id= '3952'
  and route_id= '3952';
select user_id
from m.agency
where valid_now=18161
  and agency_id_id= '14669';
select user_id
from m.agency
where valid_now=17423
  and agency_id_id= '12252';
select COUNT(*)
from dv.notes_message
where user_id='17150'
  and agency_id_id= '17150'
  and notice_id= '17150'
  and route_id= '17150';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18777'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1762
  and agency_id_id= '488';
select user_id
from m.agency
where valid_now=10291
  and agency_id_id= '7291';
select user_id
from m.agency
where valid_now=18971
  and agency_id_id= '16573';
select COUNT(*)
from dv.notes_message
where user_id='8776'
  and agency_id_id= '8776'
  and notice_id= '8776'
  and route_id= '8776';
select agency_id
from m.agency
where agency_id_id= '8229'
  and valid_now=4107;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18340'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4618'
  and agency_id_id= '4618'
  and notice_id= '4618'
  and route_id= '4618';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2097'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19721'
  and valid_now=12906;
select agency_id
from m.agency
where agency_id_id= '3150'
  and valid_now=455;
select agency_id
from m.agency
where agency_id_id= '978'
  and valid_now=19536;
select COUNT(*)
from dv.notes_message
where user_id='15043'
  and agency_id_id= '15043'
  and notice_id= '15043'
  and route_id= '15043';
select agency_id
from m.agency
where agency_id_id= '18571'
  and valid_now=19172;
select user_id
from m.agency
where valid_now=12627
  and agency_id_id= '4676';
select COUNT(*)
from dv.notes_message
where user_id='8869'
  and agency_id_id= '8869'
  and notice_id= '8869'
  and route_id= '8869';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8215'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10906'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2329
  and agency_id_id= '18348';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3154'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16690
  and agency_id_id= '1127';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19217'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5390'
  and agency_id_id= '5390'
  and notice_id= '5390'
  and route_id= '5390';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9666'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14122'
  and valid_now=4933;
select agency_id
from m.agency
where agency_id_id= '6194'
  and valid_now=3184;
select COUNT(*)
from dv.notes_message
where user_id='19861'
  and agency_id_id= '19861'
  and notice_id= '19861'
  and route_id= '19861';
select COUNT(*)
from dv.notes_message
where user_id='4739'
  and agency_id_id= '4739'
  and notice_id= '4739'
  and route_id= '4739';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15785'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19602
  and agency_id_id= '3864';
select COUNT(*)
from dv.notes_message
where user_id='7718'
  and agency_id_id= '7718'
  and notice_id= '7718'
  and route_id= '7718';
select COUNT(*)
from dv.notes_message
where user_id='7265'
  and agency_id_id= '7265'
  and notice_id= '7265'
  and route_id= '7265';
select COUNT(*)
from dv.notes_message
where user_id='524'
  and agency_id_id= '524'
  and notice_id= '524'
  and route_id= '524';
select COUNT(*)
from dv.notes_message
where user_id='9206'
  and agency_id_id= '9206'
  and notice_id= '9206'
  and route_id= '9206';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '538'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18321'
  and valid_now=7826;
select agency_id
from m.agency
where agency_id_id= '870'
  and valid_now=1601;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2782'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=17895
  and agency_id_id= '11344';
select user_id
from m.agency
where valid_now=14385
  and agency_id_id= '14886';
select COUNT(*)
from dv.notes_message
where user_id='12237'
  and agency_id_id= '12237'
  and notice_id= '12237'
  and route_id= '12237';
select COUNT(*)
from dv.notes_message
where user_id='18853'
  and agency_id_id= '18853'
  and notice_id= '18853'
  and route_id= '18853';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19604'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '786'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8991'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9737'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1295'
  and agency_id_id= '1295'
  and notice_id= '1295'
  and route_id= '1295';
select COUNT(*)
from dv.notes_message
where user_id='17051'
  and agency_id_id= '17051'
  and notice_id= '17051'
  and route_id= '17051';
select user_id
from m.agency
where valid_now=17393
  and agency_id_id= '1309';
select user_id
from m.agency
where valid_now=5664
  and agency_id_id= '15073';
select COUNT(*)
from dv.notes_message
where user_id='2125'
  and agency_id_id= '2125'
  and notice_id= '2125'
  and route_id= '2125';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5141'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12632'
  and agency_id_id= '12632'
  and notice_id= '12632'
  and route_id= '12632';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6197'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1751'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5896';
select a.agency_timezone
from m.agency a
where a.agency_id = '4509';
select COUNT(*)
from dv.notes_message
where user_id='5972'
  and agency_id_id= '5972'
  and notice_id= '5972'
  and route_id= '5972';
select COUNT(*)
from dv.notes_message
where user_id='15638'
  and agency_id_id= '15638'
  and notice_id= '15638'
  and route_id= '15638';
select user_id
from m.agency
where valid_now=8329
  and agency_id_id= '17378';
select user_id
from m.agency
where valid_now=6964
  and agency_id_id= '6838';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9558'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7207
  and agency_id_id= '4462';
select user_id
from m.agency
where valid_now=1859
  and agency_id_id= '9307';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14565'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19830'
  and valid_now=18849;
select user_id
from m.agency
where valid_now=11384
  and agency_id_id= '3613';
select agency_id
from m.agency
where agency_id_id= '2609'
  and valid_now=9749;
select COUNT(*)
from dv.notes_message
where user_id='4113'
  and agency_id_id= '4113'
  and notice_id= '4113'
  and route_id= '4113';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12024'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3859
  and agency_id_id= '467';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13244'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9293'
  and valid_now=13034;
select user_id
from m.agency
where valid_now=3992
  and agency_id_id= '5077';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14872'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5646'
  and agency_id_id= '5646'
  and notice_id= '5646'
  and route_id= '5646';
select COUNT(*)
from dv.notes_message
where user_id='18678'
  and agency_id_id= '18678'
  and notice_id= '18678'
  and route_id= '18678';
select agency_id
from m.agency
where agency_id_id= '11449'
  and valid_now=19174;
select COUNT(*)
from dv.notes_message
where user_id='18218'
  and agency_id_id= '18218'
  and notice_id= '18218'
  and route_id= '18218';
select agency_id
from m.agency
where agency_id_id= '5845'
  and valid_now=4344;
select user_id
from m.agency
where valid_now=3489
  and agency_id_id= '8392';
select COUNT(*)
from dv.notes_message
where user_id='19580'
  and agency_id_id= '19580'
  and notice_id= '19580'
  and route_id= '19580';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11004'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9703'
  and agency_id_id= '9703'
  and notice_id= '9703'
  and route_id= '9703';
select agency_id
from m.agency
where agency_id_id= '18020'
  and valid_now=6532;
select agency_id
from m.agency
where agency_id_id= '8063'
  and valid_now=5763;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6480'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4154'
  and valid_now=19131;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14919'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14771'
  and valid_now=10753;
select agency_id
from m.agency
where agency_id_id= '4493'
  and valid_now=8475;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12258'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16785'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1506
  and agency_id_id= '816';
select COUNT(*)
from dv.notes_message
where user_id='9507'
  and agency_id_id= '9507'
  and notice_id= '9507'
  and route_id= '9507';
select COUNT(*)
from dv.notes_message
where user_id='1100'
  and agency_id_id= '1100'
  and notice_id= '1100'
  and route_id= '1100';
select COUNT(*)
from dv.notes_message
where user_id='5706'
  and agency_id_id= '5706'
  and notice_id= '5706'
  and route_id= '5706';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5334'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2664'
  and valid_now=5397;
select COUNT(*)
from dv.notes_message
where user_id='11686'
  and agency_id_id= '11686'
  and notice_id= '11686'
  and route_id= '11686';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14591'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16381'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9752
  and agency_id_id= '7214';
select COUNT(*)
from dv.notes_message
where user_id='5131'
  and agency_id_id= '5131'
  and notice_id= '5131'
  and route_id= '5131';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10544'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4271
  and agency_id_id= '5351';
select agency_id
from m.agency
where agency_id_id= '3798'
  and valid_now=10826;
select COUNT(*)
from dv.notes_message
where user_id='14553'
  and agency_id_id= '14553'
  and notice_id= '14553'
  and route_id= '14553';
select COUNT(*)
from dv.notes_message
where user_id='15950'
  and agency_id_id= '15950'
  and notice_id= '15950'
  and route_id= '15950';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9632'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18368'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12732'
  and valid_now=6709;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4425'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4647'
  and agency_id_id= '4647'
  and notice_id= '4647'
  and route_id= '4647';
select COUNT(*)
from dv.notes_message
where user_id='12556'
  and agency_id_id= '12556'
  and notice_id= '12556'
  and route_id= '12556';
select agency_id
from m.agency
where agency_id_id= '19295'
  and valid_now=6856;
select user_id
from m.agency
where valid_now=14629
  and agency_id_id= '18479';
select user_id
from m.agency
where valid_now=10680
  and agency_id_id= '18468';
select agency_id
from m.agency
where agency_id_id= '10142'
  and valid_now=10530;
select user_id
from m.agency
where valid_now=7796
  and agency_id_id= '8062';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15435'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14459
  and agency_id_id= '17082';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9381'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7119'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1330
  and agency_id_id= '17156';
select COUNT(*)
from dv.notes_message
where user_id='73'
  and agency_id_id= '73'
  and notice_id= '73'
  and route_id= '73';
select COUNT(*)
from dv.notes_message
where user_id='15336'
  and agency_id_id= '15336'
  and notice_id= '15336'
  and route_id= '15336';
select user_id
from m.agency
where valid_now=9069
  and agency_id_id= '13839';
select agency_id
from m.agency
where agency_id_id= '832'
  and valid_now=18734;
select user_id
from m.agency
where valid_now=10241
  and agency_id_id= '11493';
select agency_id
from m.agency
where agency_id_id= '12336'
  and valid_now=2641;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1181'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2271'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '4842';
select user_id
from m.agency
where valid_now=19487
  and agency_id_id= '15321';
select COUNT(*)
from dv.notes_message
where user_id='7305'
  and agency_id_id= '7305'
  and notice_id= '7305'
  and route_id= '7305';
select COUNT(*)
from dv.notes_message
where user_id='17917'
  and agency_id_id= '17917'
  and notice_id= '17917'
  and route_id= '17917';
select COUNT(*)
from dv.notes_message
where user_id='13113'
  and agency_id_id= '13113'
  and notice_id= '13113'
  and route_id= '13113';
select a.agency_timezone
from m.agency a
where a.agency_id = '2933';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11182'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17670'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2755'
  and valid_now=270;
select COUNT(*)
from dv.notes_message
where user_id='4571'
  and agency_id_id= '4571'
  and notice_id= '4571'
  and route_id= '4571';
select agency_id
from m.agency
where agency_id_id= '7360'
  and valid_now=3750;
select COUNT(*)
from dv.notes_message
where user_id='4905'
  and agency_id_id= '4905'
  and notice_id= '4905'
  and route_id= '4905';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18761'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '7383';
select user_id
from m.agency
where valid_now=16437
  and agency_id_id= '1938';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6150'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13443'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18107'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '3262';
select COUNT(*)
from dv.notes_message
where user_id='19994'
  and agency_id_id= '19994'
  and notice_id= '19994'
  and route_id= '19994';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3926'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10239'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5223
  and agency_id_id= '16677';
select user_id
from m.agency
where valid_now=15775
  and agency_id_id= '6882';
select user_id
from m.agency
where valid_now=7002
  and agency_id_id= '105';
select user_id
from m.agency
where valid_now=17239
  and agency_id_id= '10153';
select COUNT(*)
from dv.notes_message
where user_id='18046'
  and agency_id_id= '18046'
  and notice_id= '18046'
  and route_id= '18046';
select user_id
from m.agency
where valid_now=471
  and agency_id_id= '14507';
select user_id
from m.agency
where valid_now=8700
  and agency_id_id= '970';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14454'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1151
  and agency_id_id= '4840';
select agency_id
from m.agency
where agency_id_id= '13199'
  and valid_now=11623;
select user_id
from m.agency
where valid_now=18286
  and agency_id_id= '1877';
select COUNT(*)
from dv.notes_message
where user_id='11451'
  and agency_id_id= '11451'
  and notice_id= '11451'
  and route_id= '11451';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '959'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1982
  and agency_id_id= '11454';
select user_id
from m.agency
where valid_now=6617
  and agency_id_id= '7925';
select agency_id
from m.agency
where agency_id_id= '2932'
  and valid_now=6026;
select COUNT(*)
from dv.notes_message
where user_id='6490'
  and agency_id_id= '6490'
  and notice_id= '6490'
  and route_id= '6490';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15537'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15511'
  and valid_now=5779;
select agency_id
from m.agency
where agency_id_id= '2775'
  and valid_now=643;
select agency_id
from m.agency
where agency_id_id= '4227'
  and valid_now=11607;
select COUNT(*)
from dv.notes_message
where user_id='13743'
  and agency_id_id= '13743'
  and notice_id= '13743'
  and route_id= '13743';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18734'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11299
  and agency_id_id= '5408';
select user_id
from m.agency
where valid_now=19256
  and agency_id_id= '13450';
select user_id
from m.agency
where valid_now=8460
  and agency_id_id= '12582';
select user_id
from m.agency
where valid_now=19182
  and agency_id_id= '4566';
select user_id
from m.agency
where valid_now=6371
  and agency_id_id= '9904';
select COUNT(*)
from dv.notes_message
where user_id='18373'
  and agency_id_id= '18373'
  and notice_id= '18373'
  and route_id= '18373';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '588'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '15306';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18840'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10272
  and agency_id_id= '14668';
select user_id
from m.agency
where valid_now=11187
  and agency_id_id= '6197';
select user_id
from m.agency
where valid_now=4553
  and agency_id_id= '12943';
select COUNT(*)
from dv.notes_message
where user_id='11428'
  and agency_id_id= '11428'
  and notice_id= '11428'
  and route_id= '11428';
select COUNT(*)
from dv.notes_message
where user_id='9539'
  and agency_id_id= '9539'
  and notice_id= '9539'
  and route_id= '9539';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9467'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17548'
  and valid_now=19184;
select agency_id
from m.agency
where agency_id_id= '11890'
  and valid_now=6656;
select agency_id
from m.agency
where agency_id_id= '6773'
  and valid_now=5269;
select user_id
from m.agency
where valid_now=15149
  and agency_id_id= '8';
select COUNT(*)
from dv.notes_message
where user_id='8875'
  and agency_id_id= '8875'
  and notice_id= '8875'
  and route_id= '8875';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3474'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '876';
select user_id
from m.agency
where valid_now=2548
  and agency_id_id= '2163';
select COUNT(*)
from dv.notes_message
where user_id='1807'
  and agency_id_id= '1807'
  and notice_id= '1807'
  and route_id= '1807';
select a.agency_timezone
from m.agency a
where a.agency_id = '2125';
select user_id
from m.agency
where valid_now=2522
  and agency_id_id= '9005';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17382'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '590'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8183
  and agency_id_id= '7350';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19317'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19303
  and agency_id_id= '1591';
select COUNT(*)
from dv.notes_message
where user_id='5734'
  and agency_id_id= '5734'
  and notice_id= '5734'
  and route_id= '5734';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6100'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7816'
  and valid_now=9732;
select agency_id
from m.agency
where agency_id_id= '14102'
  and valid_now=12961;
select agency_id
from m.agency
where agency_id_id= '14339'
  and valid_now=18855;
select COUNT(*)
from dv.notes_message
where user_id='2900'
  and agency_id_id= '2900'
  and notice_id= '2900'
  and route_id= '2900';
select agency_id
from m.agency
where agency_id_id= '4464'
  and valid_now=11902;
select agency_id
from m.agency
where agency_id_id= '3697'
  and valid_now=4551;
select agency_id
from m.agency
where agency_id_id= '13101'
  and valid_now=13590;
select COUNT(*)
from dv.notes_message
where user_id='4601'
  and agency_id_id= '4601'
  and notice_id= '4601'
  and route_id= '4601';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18335'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7183'
  and valid_now=6069;
select COUNT(*)
from dv.notes_message
where user_id='13557'
  and agency_id_id= '13557'
  and notice_id= '13557'
  and route_id= '13557';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11101'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4402'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5040';
select agency_id
from m.agency
where agency_id_id= '8156'
  and valid_now=7041;
select COUNT(*)
from dv.notes_message
where user_id='8133'
  and agency_id_id= '8133'
  and notice_id= '8133'
  and route_id= '8133';
select a.agency_timezone
from m.agency a
where a.agency_id = '3259';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9515'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19489'
  and agency_id_id= '19489'
  and notice_id= '19489'
  and route_id= '19489';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7551'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10358'
  and valid_now=9536;
select agency_id
from m.agency
where agency_id_id= '19372'
  and valid_now=532;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10214'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1192'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18699'
  and valid_now=17483;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4420'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2850
  and agency_id_id= '14337';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12869'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10160'
  and valid_now=19579;
select agency_id
from m.agency
where agency_id_id= '8475'
  and valid_now=15693;
select COUNT(*)
from dv.notes_message
where user_id='2897'
  and agency_id_id= '2897'
  and notice_id= '2897'
  and route_id= '2897';
select user_id
from m.agency
where valid_now=5181
  and agency_id_id= '17597';
select user_id
from m.agency
where valid_now=12063
  and agency_id_id= '4447';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8777'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12491'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4720'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8454'
  and valid_now=583;
select COUNT(*)
from dv.notes_message
where user_id='11124'
  and agency_id_id= '11124'
  and notice_id= '11124'
  and route_id= '11124';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2330'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14645'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9246'
  and agency_id_id= '9246'
  and notice_id= '9246'
  and route_id= '9246';
select COUNT(*)
from dv.notes_message
where user_id='7526'
  and agency_id_id= '7526'
  and notice_id= '7526'
  and route_id= '7526';
select agency_id
from m.agency
where agency_id_id= '19416'
  and valid_now=15912;
select user_id
from m.agency
where valid_now=15340
  and agency_id_id= '2402';
select agency_id
from m.agency
where agency_id_id= '600'
  and valid_now=7172;
select COUNT(*)
from dv.notes_message
where user_id='17888'
  and agency_id_id= '17888'
  and notice_id= '17888'
  and route_id= '17888';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14967'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10899'
  and valid_now=869;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17165'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5635'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '3283';
select a.agency_timezone
from m.agency a
where a.agency_id = '19208';
select COUNT(*)
from dv.notes_message
where user_id='195'
  and agency_id_id= '195'
  and notice_id= '195'
  and route_id= '195';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3600'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3713'
  and agency_id_id= '3713'
  and notice_id= '3713'
  and route_id= '3713';
select agency_id
from m.agency
where agency_id_id= '13448'
  and valid_now=16681;
select user_id
from m.agency
where valid_now=19926
  and agency_id_id= '6227';
select COUNT(*)
from dv.notes_message
where user_id='16871'
  and agency_id_id= '16871'
  and notice_id= '16871'
  and route_id= '16871';
select COUNT(*)
from dv.notes_message
where user_id='13335'
  and agency_id_id= '13335'
  and notice_id= '13335'
  and route_id= '13335';
select COUNT(*)
from dv.notes_message
where user_id='10364'
  and agency_id_id= '10364'
  and notice_id= '10364'
  and route_id= '10364';
select agency_id
from m.agency
where agency_id_id= '5782'
  and valid_now=19359;
select agency_id
from m.agency
where agency_id_id= '5119'
  and valid_now=4790;
select user_id
from m.agency
where valid_now=15911
  and agency_id_id= '17818';
select user_id
from m.agency
where valid_now=4837
  and agency_id_id= '9103';
select COUNT(*)
from dv.notes_message
where user_id='19498'
  and agency_id_id= '19498'
  and notice_id= '19498'
  and route_id= '19498';
select COUNT(*)
from dv.notes_message
where user_id='92'
  and agency_id_id= '92'
  and notice_id= '92'
  and route_id= '92';
select user_id
from m.agency
where valid_now=1622
  and agency_id_id= '13208';
select user_id
from m.agency
where valid_now=96
  and agency_id_id= '12536';
select COUNT(*)
from dv.notes_message
where user_id='891'
  and agency_id_id= '891'
  and notice_id= '891'
  and route_id= '891';
select agency_id
from m.agency
where agency_id_id= '8090'
  and valid_now=12102;
select user_id
from m.agency
where valid_now=6754
  and agency_id_id= '16453';
select COUNT(*)
from dv.notes_message
where user_id='17047'
  and agency_id_id= '17047'
  and notice_id= '17047'
  and route_id= '17047';
select user_id
from m.agency
where valid_now=16398
  and agency_id_id= '6161';
select COUNT(*)
from dv.notes_message
where user_id='18682'
  and agency_id_id= '18682'
  and notice_id= '18682'
  and route_id= '18682';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15933'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9170'
  and valid_now=10452;
select user_id
from m.agency
where valid_now=3217
  and agency_id_id= '8946';
select user_id
from m.agency
where valid_now=434
  and agency_id_id= '16759';
select user_id
from m.agency
where valid_now=10393
  and agency_id_id= '1102';
select user_id
from m.agency
where valid_now=16753
  and agency_id_id= '1879';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9764'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7847'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7091'
  and valid_now=9947;
select agency_id
from m.agency
where agency_id_id= '12826'
  and valid_now=4549;
select user_id
from m.agency
where valid_now=5730
  and agency_id_id= '13431';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4049'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17329'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1633'
  and agency_id_id= '1633'
  and notice_id= '1633'
  and route_id= '1633';
select agency_id
from m.agency
where agency_id_id= '12051'
  and valid_now=7558;
select user_id
from m.agency
where valid_now=8993
  and agency_id_id= '12577';
select user_id
from m.agency
where valid_now=6447
  and agency_id_id= '11638';
select COUNT(*)
from dv.notes_message
where user_id='19494'
  and agency_id_id= '19494'
  and notice_id= '19494'
  and route_id= '19494';
select COUNT(*)
from dv.notes_message
where user_id='19659'
  and agency_id_id= '19659'
  and notice_id= '19659'
  and route_id= '19659';
select user_id
from m.agency
where valid_now=17203
  and agency_id_id= '2761';
select COUNT(*)
from dv.notes_message
where user_id='3579'
  and agency_id_id= '3579'
  and notice_id= '3579'
  and route_id= '3579';
select agency_id
from m.agency
where agency_id_id= '5261'
  and valid_now=16991;
select user_id
from m.agency
where valid_now=15278
  and agency_id_id= '3500';
select COUNT(*)
from dv.notes_message
where user_id='19442'
  and agency_id_id= '19442'
  and notice_id= '19442'
  and route_id= '19442';
select COUNT(*)
from dv.notes_message
where user_id='3248'
  and agency_id_id= '3248'
  and notice_id= '3248'
  and route_id= '3248';
select agency_id
from m.agency
where agency_id_id= '13962'
  and valid_now=8214;
select user_id
from m.agency
where valid_now=4592
  and agency_id_id= '4807';
select agency_id
from m.agency
where agency_id_id= '16044'
  and valid_now=6523;
select COUNT(*)
from dv.notes_message
where user_id='8382'
  and agency_id_id= '8382'
  and notice_id= '8382'
  and route_id= '8382';
select agency_id
from m.agency
where agency_id_id= '1538'
  and valid_now=5972;
select COUNT(*)
from dv.notes_message
where user_id='17913'
  and agency_id_id= '17913'
  and notice_id= '17913'
  and route_id= '17913';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11063'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18931'
  and valid_now=17367;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10716'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8635'
  and valid_now=2121;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10783'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12465
  and agency_id_id= '653';
select user_id
from m.agency
where valid_now=16515
  and agency_id_id= '1983';
select user_id
from m.agency
where valid_now=5230
  and agency_id_id= '1027';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4442'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14725'
  and valid_now=19460;
select agency_id
from m.agency
where agency_id_id= '8061'
  and valid_now=9687;
select user_id
from m.agency
where valid_now=8029
  and agency_id_id= '13489';
select a.agency_timezone
from m.agency a
where a.agency_id = '12327';
select user_id
from m.agency
where valid_now=12510
  and agency_id_id= '16889';
select COUNT(*)
from dv.notes_message
where user_id='15889'
  and agency_id_id= '15889'
  and notice_id= '15889'
  and route_id= '15889';
select COUNT(*)
from dv.notes_message
where user_id='1580'
  and agency_id_id= '1580'
  and notice_id= '1580'
  and route_id= '1580';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11829'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '4863';
select COUNT(*)
from dv.notes_message
where user_id='9747'
  and agency_id_id= '9747'
  and notice_id= '9747'
  and route_id= '9747';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19336'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16824'
  and valid_now=11170;
select COUNT(*)
from dv.notes_message
where user_id='9739'
  and agency_id_id= '9739'
  and notice_id= '9739'
  and route_id= '9739';
select COUNT(*)
from dv.notes_message
where user_id='2450'
  and agency_id_id= '2450'
  and notice_id= '2450'
  and route_id= '2450';
select agency_id
from m.agency
where agency_id_id= '12304'
  and valid_now=18269;
select COUNT(*)
from dv.notes_message
where user_id='9364'
  and agency_id_id= '9364'
  and notice_id= '9364'
  and route_id= '9364';
select COUNT(*)
from dv.notes_message
where user_id='13220'
  and agency_id_id= '13220'
  and notice_id= '13220'
  and route_id= '13220';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12862'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3821
  and agency_id_id= '4902';
select agency_id
from m.agency
where agency_id_id= '6821'
  and valid_now=7355;
select agency_id
from m.agency
where agency_id_id= '11608'
  and valid_now=12355;
select user_id
from m.agency
where valid_now=16956
  and agency_id_id= '12624';
select COUNT(*)
from dv.notes_message
where user_id='2998'
  and agency_id_id= '2998'
  and notice_id= '2998'
  and route_id= '2998';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11350'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3138
  and agency_id_id= '1620';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18982'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '835'
  and valid_now=7344;
select agency_id
from m.agency
where agency_id_id= '7237'
  and valid_now=7659;
select COUNT(*)
from dv.notes_message
where user_id='19309'
  and agency_id_id= '19309'
  and notice_id= '19309'
  and route_id= '19309';
select agency_id
from m.agency
where agency_id_id= '6266'
  and valid_now=17313;
select user_id
from m.agency
where valid_now=2656
  and agency_id_id= '4078';
select user_id
from m.agency
where valid_now=1455
  and agency_id_id= '7264';
select agency_id
from m.agency
where agency_id_id= '19448'
  and valid_now=12138;
select agency_id
from m.agency
where agency_id_id= '1589'
  and valid_now=5414;
select agency_id
from m.agency
where agency_id_id= '17830'
  and valid_now=5355;
select agency_id
from m.agency
where agency_id_id= '19508'
  and valid_now=18622;
select user_id
from m.agency
where valid_now=10836
  and agency_id_id= '621';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15269'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8789'
  and valid_now=4715;
select agency_id
from m.agency
where agency_id_id= '6671'
  and valid_now=10162;
select agency_id
from m.agency
where agency_id_id= '14987'
  and valid_now=8218;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13425'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9448'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13725'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8368'
  and valid_now=2496;
select agency_id
from m.agency
where agency_id_id= '15915'
  and valid_now=18029;
select user_id
from m.agency
where valid_now=12342
  and agency_id_id= '19383';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6759'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1034'
  and valid_now=9272;
select COUNT(*)
from dv.notes_message
where user_id='14358'
  and agency_id_id= '14358'
  and notice_id= '14358'
  and route_id= '14358';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18190'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18794'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1060'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10407
  and agency_id_id= '7004';
select COUNT(*)
from dv.notes_message
where user_id='16446'
  and agency_id_id= '16446'
  and notice_id= '16446'
  and route_id= '16446';
select agency_id
from m.agency
where agency_id_id= '17345'
  and valid_now=97;
select user_id
from m.agency
where valid_now=15281
  and agency_id_id= '12318';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12883'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11951'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6446'
  and valid_now=11952;
select user_id
from m.agency
where valid_now=19621
  and agency_id_id= '1437';
select user_id
from m.agency
where valid_now=4086
  and agency_id_id= '5765';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13021'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6348'
  and agency_id_id= '6348'
  and notice_id= '6348'
  and route_id= '6348';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '371'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18859
  and agency_id_id= '13017';
select agency_id
from m.agency
where agency_id_id= '17908'
  and valid_now=3477;
select user_id
from m.agency
where valid_now=15987
  and agency_id_id= '790';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7739'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '10066';
select agency_id
from m.agency
where agency_id_id= '2546'
  and valid_now=17185;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16911'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2822'
  and valid_now=18589;
select user_id
from m.agency
where valid_now=10440
  and agency_id_id= '9284';
select user_id
from m.agency
where valid_now=6616
  and agency_id_id= '4076';
select user_id
from m.agency
where valid_now=8981
  and agency_id_id= '6352';
select COUNT(*)
from dv.notes_message
where user_id='13097'
  and agency_id_id= '13097'
  and notice_id= '13097'
  and route_id= '13097';
select COUNT(*)
from dv.notes_message
where user_id='10788'
  and agency_id_id= '10788'
  and notice_id= '10788'
  and route_id= '10788';
select COUNT(*)
from dv.notes_message
where user_id='10539'
  and agency_id_id= '10539'
  and notice_id= '10539'
  and route_id= '10539';
select agency_id
from m.agency
where agency_id_id= '3843'
  and valid_now=17096;
select agency_id
from m.agency
where agency_id_id= '12373'
  and valid_now=18695;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6840'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12411'
  and valid_now=4026;
select user_id
from m.agency
where valid_now=6515
  and agency_id_id= '4716';
select user_id
from m.agency
where valid_now=18089
  and agency_id_id= '17156';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10966'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10865'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16399
  and agency_id_id= '5075';
select agency_id
from m.agency
where agency_id_id= '19230'
  and valid_now=9649;
select user_id
from m.agency
where valid_now=2128
  and agency_id_id= '9088';
select user_id
from m.agency
where valid_now=8186
  and agency_id_id= '11160';
select COUNT(*)
from dv.notes_message
where user_id='75'
  and agency_id_id= '75'
  and notice_id= '75'
  and route_id= '75';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10708'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7682
  and agency_id_id= '2721';
select user_id
from m.agency
where valid_now=888
  and agency_id_id= '6161';
select COUNT(*)
from dv.notes_message
where user_id='10176'
  and agency_id_id= '10176'
  and notice_id= '10176'
  and route_id= '10176';
select user_id
from m.agency
where valid_now=6221
  and agency_id_id= '18253';
select user_id
from m.agency
where valid_now=7865
  and agency_id_id= '15843';
select COUNT(*)
from dv.notes_message
where user_id='182'
  and agency_id_id= '182'
  and notice_id= '182'
  and route_id= '182';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7206'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19009
  and agency_id_id= '3675';
select user_id
from m.agency
where valid_now=69
  and agency_id_id= '10799';
select COUNT(*)
from dv.notes_message
where user_id='10986'
  and agency_id_id= '10986'
  and notice_id= '10986'
  and route_id= '10986';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4959'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3813'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1793'
  and valid_now=14278;
select COUNT(*)
from dv.notes_message
where user_id='3492'
  and agency_id_id= '3492'
  and notice_id= '3492'
  and route_id= '3492';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9129'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9440'
  and valid_now=2495;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19252'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5175'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1662'
  and valid_now=16300;
select user_id
from m.agency
where valid_now=1908
  and agency_id_id= '5858';
select user_id
from m.agency
where valid_now=15424
  and agency_id_id= '9833';
select user_id
from m.agency
where valid_now=18297
  and agency_id_id= '4879';
select COUNT(*)
from dv.notes_message
where user_id='11481'
  and agency_id_id= '11481'
  and notice_id= '11481'
  and route_id= '11481';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18208'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11703'
  and valid_now=13295;
select agency_id
from m.agency
where agency_id_id= '9597'
  and valid_now=7611;
select COUNT(*)
from dv.notes_message
where user_id='423'
  and agency_id_id= '423'
  and notice_id= '423'
  and route_id= '423';
select COUNT(*)
from dv.notes_message
where user_id='585'
  and agency_id_id= '585'
  and notice_id= '585'
  and route_id= '585';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7742'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10478
  and agency_id_id= '19063';
select user_id
from m.agency
where valid_now=10420
  and agency_id_id= '7386';
select user_id
from m.agency
where valid_now=1184
  and agency_id_id= '19104';
select COUNT(*)
from dv.notes_message
where user_id='10807'
  and agency_id_id= '10807'
  and notice_id= '10807'
  and route_id= '10807';
select COUNT(*)
from dv.notes_message
where user_id='12372'
  and agency_id_id= '12372'
  and notice_id= '12372'
  and route_id= '12372';
select agency_id
from m.agency
where agency_id_id= '19065'
  and valid_now=16118;
select agency_id
from m.agency
where agency_id_id= '7866'
  and valid_now=6785;
select user_id
from m.agency
where valid_now=16339
  and agency_id_id= '656';
select user_id
from m.agency
where valid_now=15200
  and agency_id_id= '10559';
select COUNT(*)
from dv.notes_message
where user_id='8320'
  and agency_id_id= '8320'
  and notice_id= '8320'
  and route_id= '8320';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17076'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '288'
  and valid_now=11266;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11222'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5331
  and agency_id_id= '8150';
select COUNT(*)
from dv.notes_message
where user_id='11356'
  and agency_id_id= '11356'
  and notice_id= '11356'
  and route_id= '11356';
select COUNT(*)
from dv.notes_message
where user_id='18309'
  and agency_id_id= '18309'
  and notice_id= '18309'
  and route_id= '18309';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2411'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18209'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15426'
  and valid_now=1571;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9872'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1849
  and agency_id_id= '7742';
select user_id
from m.agency
where valid_now=7997
  and agency_id_id= '1955';
select user_id
from m.agency
where valid_now=12157
  and agency_id_id= '17101';
select COUNT(*)
from dv.notes_message
where user_id='16777'
  and agency_id_id= '16777'
  and notice_id= '16777'
  and route_id= '16777';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '164'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14456'
  and agency_id_id= '14456'
  and notice_id= '14456'
  and route_id= '14456';
select agency_id
from m.agency
where agency_id_id= '19435'
  and valid_now=16321;
select COUNT(*)
from dv.notes_message
where user_id='10073'
  and agency_id_id= '10073'
  and notice_id= '10073'
  and route_id= '10073';
select COUNT(*)
from dv.notes_message
where user_id='6279'
  and agency_id_id= '6279'
  and notice_id= '6279'
  and route_id= '6279';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16467'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19855'
  and valid_now=9202;
select user_id
from m.agency
where valid_now=6590
  and agency_id_id= '16484';
select user_id
from m.agency
where valid_now=3073
  and agency_id_id= '15807';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2467'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4385
  and agency_id_id= '5957';
select user_id
from m.agency
where valid_now=7403
  and agency_id_id= '13959';
select agency_id
from m.agency
where agency_id_id= '1503'
  and valid_now=3872;
select agency_id
from m.agency
where agency_id_id= '4421'
  and valid_now=16388;
select agency_id
from m.agency
where agency_id_id= '19101'
  and valid_now=14062;
select agency_id
from m.agency
where agency_id_id= '15089'
  and valid_now=4037;
select agency_id
from m.agency
where agency_id_id= '5107'
  and valid_now=5202;
select user_id
from m.agency
where valid_now=12635
  and agency_id_id= '19911';
select COUNT(*)
from dv.notes_message
where user_id='12929'
  and agency_id_id= '12929'
  and notice_id= '12929'
  and route_id= '12929';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '732'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11705
  and agency_id_id= '8079';
select COUNT(*)
from dv.notes_message
where user_id='9974'
  and agency_id_id= '9974'
  and notice_id= '9974'
  and route_id= '9974';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13924'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6475'
  and valid_now=10445;
select user_id
from m.agency
where valid_now=12379
  and agency_id_id= '8797';
select COUNT(*)
from dv.notes_message
where user_id='13544'
  and agency_id_id= '13544'
  and notice_id= '13544'
  and route_id= '13544';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3903'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1598
  and agency_id_id= '5349';
select COUNT(*)
from dv.notes_message
where user_id='6358'
  and agency_id_id= '6358'
  and notice_id= '6358'
  and route_id= '6358';
select COUNT(*)
from dv.notes_message
where user_id='4726'
  and agency_id_id= '4726'
  and notice_id= '4726'
  and route_id= '4726';
select agency_id
from m.agency
where agency_id_id= '11818'
  and valid_now=8904;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16099'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11084'
  and valid_now=2775;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1156'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1218'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13290'
  and valid_now=14381;
select user_id
from m.agency
where valid_now=16516
  and agency_id_id= '8545';
select user_id
from m.agency
where valid_now=14836
  and agency_id_id= '10719';
select user_id
from m.agency
where valid_now=14998
  and agency_id_id= '8528';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4357'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6369
  and agency_id_id= '19055';
select COUNT(*)
from dv.notes_message
where user_id='5304'
  and agency_id_id= '5304'
  and notice_id= '5304'
  and route_id= '5304';
select agency_id
from m.agency
where agency_id_id= '2384'
  and valid_now=11275;
select user_id
from m.agency
where valid_now=2300
  and agency_id_id= '15171';
select agency_id
from m.agency
where agency_id_id= '4098'
  and valid_now=19306;
select user_id
from m.agency
where valid_now=1538
  and agency_id_id= '14763';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6957'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16119'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9287'
  and valid_now=6204;
select user_id
from m.agency
where valid_now=11530
  and agency_id_id= '11363';
select agency_id
from m.agency
where agency_id_id= '15129'
  and valid_now=8198;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17844'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6155'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18216'
  and valid_now=12318;
select agency_id
from m.agency
where agency_id_id= '9410'
  and valid_now=6779;
select user_id
from m.agency
where valid_now=18790
  and agency_id_id= '13483';
select user_id
from m.agency
where valid_now=3732
  and agency_id_id= '7883';
select COUNT(*)
from dv.notes_message
where user_id='15531'
  and agency_id_id= '15531'
  and notice_id= '15531'
  and route_id= '15531';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9506'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6292'
  and valid_now=11086;
select agency_id
from m.agency
where agency_id_id= '18979'
  and valid_now=15874;
select agency_id
from m.agency
where agency_id_id= '6187'
  and valid_now=5726;
select COUNT(*)
from dv.notes_message
where user_id='9029'
  and agency_id_id= '9029'
  and notice_id= '9029'
  and route_id= '9029';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19184'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18480'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10290'
  and valid_now=4053;
select agency_id
from m.agency
where agency_id_id= '16529'
  and valid_now=19869;
select COUNT(*)
from dv.notes_message
where user_id='11975'
  and agency_id_id= '11975'
  and notice_id= '11975'
  and route_id= '11975';
select user_id
from m.agency
where valid_now=10128
  and agency_id_id= '10521';
select COUNT(*)
from dv.notes_message
where user_id='3803'
  and agency_id_id= '3803'
  and notice_id= '3803'
  and route_id= '3803';
select agency_id
from m.agency
where agency_id_id= '14172'
  and valid_now=18999;
select user_id
from m.agency
where valid_now=9779
  and agency_id_id= '11400';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3166'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16641'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19034'
  and agency_id_id= '19034'
  and notice_id= '19034'
  and route_id= '19034';
select user_id
from m.agency
where valid_now=2605
  and agency_id_id= '11373';
select COUNT(*)
from dv.notes_message
where user_id='9540'
  and agency_id_id= '9540'
  and notice_id= '9540'
  and route_id= '9540';
select agency_id
from m.agency
where agency_id_id= '9635'
  and valid_now=990;
select COUNT(*)
from dv.notes_message
where user_id='5169'
  and agency_id_id= '5169'
  and notice_id= '5169'
  and route_id= '5169';
select agency_id
from m.agency
where agency_id_id= '2406'
  and valid_now=7343;
select COUNT(*)
from dv.notes_message
where user_id='18637'
  and agency_id_id= '18637'
  and notice_id= '18637'
  and route_id= '18637';
select agency_id
from m.agency
where agency_id_id= '17427'
  and valid_now=7515;
select user_id
from m.agency
where valid_now=1871
  and agency_id_id= '473';
select COUNT(*)
from dv.notes_message
where user_id='12144'
  and agency_id_id= '12144'
  and notice_id= '12144'
  and route_id= '12144';
select agency_id
from m.agency
where agency_id_id= '12026'
  and valid_now=3806;
select agency_id
from m.agency
where agency_id_id= '16882'
  and valid_now=737;
select agency_id
from m.agency
where agency_id_id= '16418'
  and valid_now=14162;
select user_id
from m.agency
where valid_now=17372
  and agency_id_id= '14229';
select user_id
from m.agency
where valid_now=9046
  and agency_id_id= '19612';
select COUNT(*)
from dv.notes_message
where user_id='16000'
  and agency_id_id= '16000'
  and notice_id= '16000'
  and route_id= '16000';
select user_id
from m.agency
where valid_now=1851
  and agency_id_id= '15681';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10667'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11949'
  and valid_now=19308;
select agency_id
from m.agency
where agency_id_id= '10670'
  and valid_now=279;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19124'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8785'
  and valid_now=18207;
select user_id
from m.agency
where valid_now=3972
  and agency_id_id= '5878';
select user_id
from m.agency
where valid_now=6822
  and agency_id_id= '19393';
select COUNT(*)
from dv.notes_message
where user_id='19676'
  and agency_id_id= '19676'
  and notice_id= '19676'
  and route_id= '19676';
select agency_id
from m.agency
where agency_id_id= '3757'
  and valid_now=6048;
select user_id
from m.agency
where valid_now=7968
  and agency_id_id= '19794';
select user_id
from m.agency
where valid_now=3489
  and agency_id_id= '3475';
select COUNT(*)
from dv.notes_message
where user_id='11294'
  and agency_id_id= '11294'
  and notice_id= '11294'
  and route_id= '11294';
select agency_id
from m.agency
where agency_id_id= '16268'
  and valid_now=7145;
select user_id
from m.agency
where valid_now=11319
  and agency_id_id= '13150';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5530'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13193'
  and valid_now=10152;
select agency_id
from m.agency
where agency_id_id= '1173'
  and valid_now=9017;
select COUNT(*)
from dv.notes_message
where user_id='18969'
  and agency_id_id= '18969'
  and notice_id= '18969'
  and route_id= '18969';
select COUNT(*)
from dv.notes_message
where user_id='16651'
  and agency_id_id= '16651'
  and notice_id= '16651'
  and route_id= '16651';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16837'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7440'
  and valid_now=6695;
select user_id
from m.agency
where valid_now=17133
  and agency_id_id= '14121';
select user_id
from m.agency
where valid_now=1484
  and agency_id_id= '4239';
select agency_id
from m.agency
where agency_id_id= '5176'
  and valid_now=18752;
select COUNT(*)
from dv.notes_message
where user_id='6751'
  and agency_id_id= '6751'
  and notice_id= '6751'
  and route_id= '6751';
select COUNT(*)
from dv.notes_message
where user_id='5676'
  and agency_id_id= '5676'
  and notice_id= '5676'
  and route_id= '5676';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '977'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17748'
  and valid_now=11755;
select agency_id
from m.agency
where agency_id_id= '19457'
  and valid_now=19001;
select COUNT(*)
from dv.notes_message
where user_id='18514'
  and agency_id_id= '18514'
  and notice_id= '18514'
  and route_id= '18514';
select agency_id
from m.agency
where agency_id_id= '10780'
  and valid_now=18667;
select user_id
from m.agency
where valid_now=8409
  and agency_id_id= '5729';
select COUNT(*)
from dv.notes_message
where user_id='10219'
  and agency_id_id= '10219'
  and notice_id= '10219'
  and route_id= '10219';
select COUNT(*)
from dv.notes_message
where user_id='15027'
  and agency_id_id= '15027'
  and notice_id= '15027'
  and route_id= '15027';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6559'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14636'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16909
  and agency_id_id= '2947';
select COUNT(*)
from dv.notes_message
where user_id='12652'
  and agency_id_id= '12652'
  and notice_id= '12652'
  and route_id= '12652';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8131'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '83'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10651'
  and valid_now=13599;
select user_id
from m.agency
where valid_now=11393
  and agency_id_id= '5657';
select user_id
from m.agency
where valid_now=9613
  and agency_id_id= '5767';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5887'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15685
  and agency_id_id= '6222';
select COUNT(*)
from dv.notes_message
where user_id='3065'
  and agency_id_id= '3065'
  and notice_id= '3065'
  and route_id= '3065';
select COUNT(*)
from dv.notes_message
where user_id='12728'
  and agency_id_id= '12728'
  and notice_id= '12728'
  and route_id= '12728';
select agency_id
from m.agency
where agency_id_id= '427'
  and valid_now=15913;
select user_id
from m.agency
where valid_now=2277
  and agency_id_id= '985';
select COUNT(*)
from dv.notes_message
where user_id='14911'
  and agency_id_id= '14911'
  and notice_id= '14911'
  and route_id= '14911';
select user_id
from m.agency
where valid_now=16866
  and agency_id_id= '16389';
select COUNT(*)
from dv.notes_message
where user_id='3885'
  and agency_id_id= '3885'
  and notice_id= '3885'
  and route_id= '3885';
select user_id
from m.agency
where valid_now=8375
  and agency_id_id= '1268';
select COUNT(*)
from dv.notes_message
where user_id='9413'
  and agency_id_id= '9413'
  and notice_id= '9413'
  and route_id= '9413';
select agency_id
from m.agency
where agency_id_id= '1435'
  and valid_now=15670;
select agency_id
from m.agency
where agency_id_id= '16007'
  and valid_now=15352;
select user_id
from m.agency
where valid_now=10327
  and agency_id_id= '10945';
select user_id
from m.agency
where valid_now=1934
  and agency_id_id= '8766';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3293'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8617'
  and valid_now=4704;
select user_id
from m.agency
where valid_now=14929
  and agency_id_id= '4761';
select user_id
from m.agency
where valid_now=18006
  and agency_id_id= '7340';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14568'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10711'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4514'
  and valid_now=13407;
select user_id
from m.agency
where valid_now=14289
  and agency_id_id= '17097';
select user_id
from m.agency
where valid_now=7096
  and agency_id_id= '10164';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '644'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4831'
  and agency_id_id= '4831'
  and notice_id= '4831'
  and route_id= '4831';
select COUNT(*)
from dv.notes_message
where user_id='9001'
  and agency_id_id= '9001'
  and notice_id= '9001'
  and route_id= '9001';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4295'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1775'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5387'
  and valid_now=13784;
select agency_id
from m.agency
where agency_id_id= '6615'
  and valid_now=15834;
select user_id
from m.agency
where valid_now=11827
  and agency_id_id= '18388';
select user_id
from m.agency
where valid_now=9002
  and agency_id_id= '14893';
select COUNT(*)
from dv.notes_message
where user_id='3660'
  and agency_id_id= '3660'
  and notice_id= '3660'
  and route_id= '3660';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17123'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15840'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3948
  and agency_id_id= '8090';
select agency_id
from m.agency
where agency_id_id= '16000'
  and valid_now=2899;
select a.agency_timezone
from m.agency a
where a.agency_id = '19635';
select COUNT(*)
from dv.notes_message
where user_id='5262'
  and agency_id_id= '5262'
  and notice_id= '5262'
  and route_id= '5262';
select a.agency_timezone
from m.agency a
where a.agency_id = '16153';
select a.agency_timezone
from m.agency a
where a.agency_id = '13366';
select COUNT(*)
from dv.notes_message
where user_id='15500'
  and agency_id_id= '15500'
  and notice_id= '15500'
  and route_id= '15500';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18965'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9993
  and agency_id_id= '13064';
select COUNT(*)
from dv.notes_message
where user_id='9685'
  and agency_id_id= '9685'
  and notice_id= '9685'
  and route_id= '9685';
select user_id
from m.agency
where valid_now=17028
  and agency_id_id= '11262';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12613'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3653'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13572'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7694'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '9309';
select a.agency_timezone
from m.agency a
where a.agency_id = '3757';
select user_id
from m.agency
where valid_now=14643
  and agency_id_id= '11837';
select COUNT(*)
from dv.notes_message
where user_id='14561'
  and agency_id_id= '14561'
  and notice_id= '14561'
  and route_id= '14561';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1159'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '108'
  and valid_now=4356;
select agency_id
from m.agency
where agency_id_id= '12596'
  and valid_now=8480;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2745'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '1202';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5120'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3132'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '6352';
select user_id
from m.agency
where valid_now=18555
  and agency_id_id= '4691';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15739'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '14189';
select COUNT(*)
from dv.notes_message
where user_id='6818'
  and agency_id_id= '6818'
  and notice_id= '6818'
  and route_id= '6818';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11225'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '14643';
select a.agency_timezone
from m.agency a
where a.agency_id = '3534';
select a.agency_timezone
from m.agency a
where a.agency_id = '12555';
select a.agency_timezone
from m.agency a
where a.agency_id = '2935';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4461'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1387'
  and valid_now=11015;
select a.agency_timezone
from m.agency a
where a.agency_id = '8953';
select user_id
from m.agency
where valid_now=15462
  and agency_id_id= '5112';
select COUNT(*)
from dv.notes_message
where user_id='19274'
  and agency_id_id= '19274'
  and notice_id= '19274'
  and route_id= '19274';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10733'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1948'
  and valid_now=5493;
select agency_id
from m.agency
where agency_id_id= '531'
  and valid_now=5618;
select agency_id
from m.agency
where agency_id_id= '8802'
  and valid_now=2613;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12096'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8884'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13394'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14076'
  and valid_now=12326;
select agency_id
from m.agency
where agency_id_id= '7293'
  and valid_now=18279;
select COUNT(*)
from dv.notes_message
where user_id='9794'
  and agency_id_id= '9794'
  and notice_id= '9794'
  and route_id= '9794';
select agency_id
from m.agency
where agency_id_id= '5194'
  and valid_now=14986;
select user_id
from m.agency
where valid_now=15499
  and agency_id_id= '16258';
select COUNT(*)
from dv.notes_message
where user_id='19573'
  and agency_id_id= '19573'
  and notice_id= '19573'
  and route_id= '19573';
select user_id
from m.agency
where valid_now=2172
  and agency_id_id= '18709';
select user_id
from m.agency
where valid_now=11819
  and agency_id_id= '3058';
select COUNT(*)
from dv.notes_message
where user_id='1457'
  and agency_id_id= '1457'
  and notice_id= '1457'
  and route_id= '1457';
select COUNT(*)
from dv.notes_message
where user_id='4748'
  and agency_id_id= '4748'
  and notice_id= '4748'
  and route_id= '4748';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11185'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10448'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7894'
  and valid_now=3987;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15222'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '575'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12989
  and agency_id_id= '2902';
select user_id
from m.agency
where valid_now=11845
  and agency_id_id= '18595';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16440'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17086'
  and valid_now=8253;
select user_id
from m.agency
where valid_now=3043
  and agency_id_id= '14746';
select user_id
from m.agency
where valid_now=2780
  and agency_id_id= '19438';
select user_id
from m.agency
where valid_now=17155
  and agency_id_id= '3433';
select COUNT(*)
from dv.notes_message
where user_id='10818'
  and agency_id_id= '10818'
  and notice_id= '10818'
  and route_id= '10818';
select agency_id
from m.agency
where agency_id_id= '19577'
  and valid_now=13455;
select COUNT(*)
from dv.notes_message
where user_id='19104'
  and agency_id_id= '19104'
  and notice_id= '19104'
  and route_id= '19104';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17704'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12956'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14'
  and valid_now=11541;
select user_id
from m.agency
where valid_now=19511
  and agency_id_id= '12328';
select COUNT(*)
from dv.notes_message
where user_id='2065'
  and agency_id_id= '2065'
  and notice_id= '2065'
  and route_id= '2065';
select agency_id
from m.agency
where agency_id_id= '16935'
  and valid_now=9535;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '765'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7710'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9867
  and agency_id_id= '4860';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5360'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11725
  and agency_id_id= '12729';
select COUNT(*)
from dv.notes_message
where user_id='16310'
  and agency_id_id= '16310'
  and notice_id= '16310'
  and route_id= '16310';
select COUNT(*)
from dv.notes_message
where user_id='15429'
  and agency_id_id= '15429'
  and notice_id= '15429'
  and route_id= '15429';
select COUNT(*)
from dv.notes_message
where user_id='2062'
  and agency_id_id= '2062'
  and notice_id= '2062'
  and route_id= '2062';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19244'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '516'
  and valid_now=12839;
select user_id
from m.agency
where valid_now=6554
  and agency_id_id= '18533';
select COUNT(*)
from dv.notes_message
where user_id='6520'
  and agency_id_id= '6520'
  and notice_id= '6520'
  and route_id= '6520';
select agency_id
from m.agency
where agency_id_id= '9451'
  and valid_now=15473;
select agency_id
from m.agency
where agency_id_id= '14308'
  and valid_now=4005;
select user_id
from m.agency
where valid_now=6865
  and agency_id_id= '15878';
select user_id
from m.agency
where valid_now=501
  and agency_id_id= '7638';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16176'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14967'
  and valid_now=744;
select user_id
from m.agency
where valid_now=14492
  and agency_id_id= '4238';
select COUNT(*)
from dv.notes_message
where user_id='9805'
  and agency_id_id= '9805'
  and notice_id= '9805'
  and route_id= '9805';
select COUNT(*)
from dv.notes_message
where user_id='3499'
  and agency_id_id= '3499'
  and notice_id= '3499'
  and route_id= '3499';
select agency_id
from m.agency
where agency_id_id= '13322'
  and valid_now=5403;
select user_id
from m.agency
where valid_now=249
  and agency_id_id= '4077';
select agency_id
from m.agency
where agency_id_id= '4863'
  and valid_now=3010;
select user_id
from m.agency
where valid_now=5677
  and agency_id_id= '17819';
select a.agency_timezone
from m.agency a
where a.agency_id = '18677';
select a.agency_timezone
from m.agency a
where a.agency_id = '2861';
select a.agency_timezone
from m.agency a
where a.agency_id = '2117';
select user_id
from m.agency
where valid_now=18178
  and agency_id_id= '5408';
select user_id
from m.agency
where valid_now=7713
  and agency_id_id= '2541';
select user_id
from m.agency
where valid_now=2563
  and agency_id_id= '7626';
select a.agency_timezone
from m.agency a
where a.agency_id = '2467';
select a.agency_timezone
from m.agency a
where a.agency_id = '5406';
select agency_id
from m.agency
where agency_id_id= '18328'
  and valid_now=4905;
select a.agency_timezone
from m.agency a
where a.agency_id = '19437';
select a.agency_timezone
from m.agency a
where a.agency_id = '8281';
select a.agency_timezone
from m.agency a
where a.agency_id = '18914';
select a.agency_timezone
from m.agency a
where a.agency_id = '12132';
select user_id
from m.agency
where valid_now=7935
  and agency_id_id= '1165';
select agency_id
from m.agency
where agency_id_id= '5746'
  and valid_now=11913;
select agency_id
from m.agency
where agency_id_id= '19827'
  and valid_now=11505;
select COUNT(*)
from dv.notes_message
where user_id='7785'
  and agency_id_id= '7785'
  and notice_id= '7785'
  and route_id= '7785';
select COUNT(*)
from dv.notes_message
where user_id='9110'
  and agency_id_id= '9110'
  and notice_id= '9110'
  and route_id= '9110';
select a.agency_timezone
from m.agency a
where a.agency_id = '19224';
select user_id
from m.agency
where valid_now=6524
  and agency_id_id= '5548';
select a.agency_timezone
from m.agency a
where a.agency_id = '19023';
select agency_id
from m.agency
where agency_id_id= '13184'
  and valid_now=14029;
select user_id
from m.agency
where valid_now=8205
  and agency_id_id= '14847';
select user_id
from m.agency
where valid_now=8826
  and agency_id_id= '9457';
select COUNT(*)
from dv.notes_message
where user_id='18085'
  and agency_id_id= '18085'
  and notice_id= '18085'
  and route_id= '18085';
select a.agency_timezone
from m.agency a
where a.agency_id = '6967';
select agency_id
from m.agency
where agency_id_id= '15908'
  and valid_now=6754;
select user_id
from m.agency
where valid_now=19584
  and agency_id_id= '15598';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7799'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12972'
  and valid_now=9560;
select agency_id
from m.agency
where agency_id_id= '17701'
  and valid_now=217;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8243'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13065'
  and valid_now=3618;
select COUNT(*)
from dv.notes_message
where user_id='457'
  and agency_id_id= '457'
  and notice_id= '457'
  and route_id= '457';
select COUNT(*)
from dv.notes_message
where user_id='5941'
  and agency_id_id= '5941'
  and notice_id= '5941'
  and route_id= '5941';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14908'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17073'
  and valid_now=3256;
select user_id
from m.agency
where valid_now=4787
  and agency_id_id= '9985';
select COUNT(*)
from dv.notes_message
where user_id='15538'
  and agency_id_id= '15538'
  and notice_id= '15538'
  and route_id= '15538';
select agency_id
from m.agency
where agency_id_id= '2721'
  and valid_now=13418;
select agency_id
from m.agency
where agency_id_id= '8336'
  and valid_now=869;
select user_id
from m.agency
where valid_now=4906
  and agency_id_id= '10565';
select COUNT(*)
from dv.notes_message
where user_id='11823'
  and agency_id_id= '11823'
  and notice_id= '11823'
  and route_id= '11823';
select COUNT(*)
from dv.notes_message
where user_id='14463'
  and agency_id_id= '14463'
  and notice_id= '14463'
  and route_id= '14463';
select COUNT(*)
from dv.notes_message
where user_id='4865'
  and agency_id_id= '4865'
  and notice_id= '4865'
  and route_id= '4865';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19118'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5968'
  and valid_now=17240;
select agency_id
from m.agency
where agency_id_id= '17735'
  and valid_now=13957;
select user_id
from m.agency
where valid_now=10635
  and agency_id_id= '14723';
select COUNT(*)
from dv.notes_message
where user_id='5516'
  and agency_id_id= '5516'
  and notice_id= '5516'
  and route_id= '5516';
select agency_id
from m.agency
where agency_id_id= '7424'
  and valid_now=11009;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11677'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2358
  and agency_id_id= '18888';
select user_id
from m.agency
where valid_now=19736
  and agency_id_id= '14059';
select COUNT(*)
from dv.notes_message
where user_id='5108'
  and agency_id_id= '5108'
  and notice_id= '5108'
  and route_id= '5108';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9889'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14513'
  and valid_now=8820;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4270'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=718
  and agency_id_id= '14369';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9982'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1598
  and agency_id_id= '7730';
select COUNT(*)
from dv.notes_message
where user_id='19325'
  and agency_id_id= '19325'
  and notice_id= '19325'
  and route_id= '19325';
select agency_id
from m.agency
where agency_id_id= '10952'
  and valid_now=17720;
select COUNT(*)
from dv.notes_message
where user_id='3089'
  and agency_id_id= '3089'
  and notice_id= '3089'
  and route_id= '3089';
select agency_id
from m.agency
where agency_id_id= '17778'
  and valid_now=4904;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6547'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15673
  and agency_id_id= '8232';
select agency_id
from m.agency
where agency_id_id= '9501'
  and valid_now=17937;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9936'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '71'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8035'
  and valid_now=11926;
select agency_id
from m.agency
where agency_id_id= '2626'
  and valid_now=11736;
select COUNT(*)
from dv.notes_message
where user_id='12276'
  and agency_id_id= '12276'
  and notice_id= '12276'
  and route_id= '12276';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4046'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1625'
  and valid_now=12444;
select agency_id
from m.agency
where agency_id_id= '8719'
  and valid_now=14499;
select user_id
from m.agency
where valid_now=12779
  and agency_id_id= '708';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7944'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9785
  and agency_id_id= '9836';
select COUNT(*)
from dv.notes_message
where user_id='9307'
  and agency_id_id= '9307'
  and notice_id= '9307'
  and route_id= '9307';
select COUNT(*)
from dv.notes_message
where user_id='3263'
  and agency_id_id= '3263'
  and notice_id= '3263'
  and route_id= '3263';
select agency_id
from m.agency
where agency_id_id= '15224'
  and valid_now=5413;
select agency_id
from m.agency
where agency_id_id= '13593'
  and valid_now=5802;
select agency_id
from m.agency
where agency_id_id= '7947'
  and valid_now=12425;
select agency_id
from m.agency
where agency_id_id= '2155'
  and valid_now=16332;
select agency_id
from m.agency
where agency_id_id= '19502'
  and valid_now=13437;
select agency_id
from m.agency
where agency_id_id= '5506'
  and valid_now=14821;
select user_id
from m.agency
where valid_now=13094
  and agency_id_id= '18534';
select user_id
from m.agency
where valid_now=8296
  and agency_id_id= '19408';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12142'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18034'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13470'
  and valid_now=13913;
select user_id
from m.agency
where valid_now=3892
  and agency_id_id= '926';
select user_id
from m.agency
where valid_now=1296
  and agency_id_id= '11435';
select user_id
from m.agency
where valid_now=10957
  and agency_id_id= '4152';
select agency_id
from m.agency
where agency_id_id= '2964'
  and valid_now=2744;
select user_id
from m.agency
where valid_now=5148
  and agency_id_id= '14102';
select user_id
from m.agency
where valid_now=2337
  and agency_id_id= '5685';
select COUNT(*)
from dv.notes_message
where user_id='14744'
  and agency_id_id= '14744'
  and notice_id= '14744'
  and route_id= '14744';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7965'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5186
  and agency_id_id= '14353';
select user_id
from m.agency
where valid_now=10469
  and agency_id_id= '5305';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11990'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7190'
  and valid_now=4690;
select agency_id
from m.agency
where agency_id_id= '19859'
  and valid_now=7870;
select user_id
from m.agency
where valid_now=14579
  and agency_id_id= '1583';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1270'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5086'
  and valid_now=19168;
select COUNT(*)
from dv.notes_message
where user_id='18240'
  and agency_id_id= '18240'
  and notice_id= '18240'
  and route_id= '18240';
select agency_id
from m.agency
where agency_id_id= '13891'
  and valid_now=15807;
select user_id
from m.agency
where valid_now=550
  and agency_id_id= '5850';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3797'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10902'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1547
  and agency_id_id= '6250';
select COUNT(*)
from dv.notes_message
where user_id='18714'
  and agency_id_id= '18714'
  and notice_id= '18714'
  and route_id= '18714';
select user_id
from m.agency
where valid_now=2284
  and agency_id_id= '7156';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2868'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8621
  and agency_id_id= '8877';
select COUNT(*)
from dv.notes_message
where user_id='15015'
  and agency_id_id= '15015'
  and notice_id= '15015'
  and route_id= '15015';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19714'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6881'
  and agency_id_id= '6881'
  and notice_id= '6881'
  and route_id= '6881';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3584'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13233'
  and agency_id_id= '13233'
  and notice_id= '13233'
  and route_id= '13233';
select agency_id
from m.agency
where agency_id_id= '4059'
  and valid_now=14404;
select agency_id
from m.agency
where agency_id_id= '3592'
  and valid_now=19061;
select user_id
from m.agency
where valid_now=4197
  and agency_id_id= '12783';
select user_id
from m.agency
where valid_now=816
  and agency_id_id= '7050';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19231'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '322'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5449'
  and valid_now=6011;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10666'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2556
  and agency_id_id= '4441';
select user_id
from m.agency
where valid_now=3844
  and agency_id_id= '10455';
select agency_id
from m.agency
where agency_id_id= '6715'
  and valid_now=12844;
select agency_id
from m.agency
where agency_id_id= '13315'
  and valid_now=9840;
select user_id
from m.agency
where valid_now=8430
  and agency_id_id= '9342';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13097'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14030'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17520'
  and valid_now=8107;
select agency_id
from m.agency
where agency_id_id= '5610'
  and valid_now=3309;
select agency_id
from m.agency
where agency_id_id= '15974'
  and valid_now=5108;
select agency_id
from m.agency
where agency_id_id= '12327'
  and valid_now=19651;
select agency_id
from m.agency
where agency_id_id= '16000'
  and valid_now=5877;
select user_id
from m.agency
where valid_now=12616
  and agency_id_id= '14908';
select user_id
from m.agency
where valid_now=5339
  and agency_id_id= '9049';
select agency_id
from m.agency
where agency_id_id= '11822'
  and valid_now=13575;
select user_id
from m.agency
where valid_now=130
  and agency_id_id= '1919';
select user_id
from m.agency
where valid_now=11470
  and agency_id_id= '17972';
select user_id
from m.agency
where valid_now=6949
  and agency_id_id= '13805';
select agency_id
from m.agency
where agency_id_id= '18399'
  and valid_now=18022;
select user_id
from m.agency
where valid_now=18284
  and agency_id_id= '7539';
select user_id
from m.agency
where valid_now=11220
  and agency_id_id= '5482';
select user_id
from m.agency
where valid_now=9880
  and agency_id_id= '6083';
select COUNT(*)
from dv.notes_message
where user_id='6485'
  and agency_id_id= '6485'
  and notice_id= '6485'
  and route_id= '6485';
select COUNT(*)
from dv.notes_message
where user_id='3877'
  and agency_id_id= '3877'
  and notice_id= '3877'
  and route_id= '3877';
select COUNT(*)
from dv.notes_message
where user_id='18959'
  and agency_id_id= '18959'
  and notice_id= '18959'
  and route_id= '18959';
select user_id
from m.agency
where valid_now=830
  and agency_id_id= '18005';
select agency_id
from m.agency
where agency_id_id= '2742'
  and valid_now=14317;
select agency_id
from m.agency
where agency_id_id= '6447'
  and valid_now=9064;
select user_id
from m.agency
where valid_now=9437
  and agency_id_id= '5263';
select COUNT(*)
from dv.notes_message
where user_id='5350'
  and agency_id_id= '5350'
  and notice_id= '5350'
  and route_id= '5350';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14447'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11124'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13013'
  and valid_now=5304;
select COUNT(*)
from dv.notes_message
where user_id='14184'
  and agency_id_id= '14184'
  and notice_id= '14184'
  and route_id= '14184';
select COUNT(*)
from dv.notes_message
where user_id='11335'
  and agency_id_id= '11335'
  and notice_id= '11335'
  and route_id= '11335';
select COUNT(*)
from dv.notes_message
where user_id='9089'
  and agency_id_id= '9089'
  and notice_id= '9089'
  and route_id= '9089';
select user_id
from m.agency
where valid_now=3818
  and agency_id_id= '17428';
select agency_id
from m.agency
where agency_id_id= '7605'
  and valid_now=1977;
select COUNT(*)
from dv.notes_message
where user_id='4773'
  and agency_id_id= '4773'
  and notice_id= '4773'
  and route_id= '4773';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6967'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11777'
  and valid_now=5892;
select COUNT(*)
from dv.notes_message
where user_id='5812'
  and agency_id_id= '5812'
  and notice_id= '5812'
  and route_id= '5812';
select agency_id
from m.agency
where agency_id_id= '10440'
  and valid_now=10032;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6782'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '454'
  and valid_now=19669;
select agency_id
from m.agency
where agency_id_id= '8962'
  and valid_now=4539;
select agency_id
from m.agency
where agency_id_id= '16075'
  and valid_now=19424;
select COUNT(*)
from dv.notes_message
where user_id='9640'
  and agency_id_id= '9640'
  and notice_id= '9640'
  and route_id= '9640';
select agency_id
from m.agency
where agency_id_id= '5056'
  and valid_now=5483;
select agency_id
from m.agency
where agency_id_id= '11732'
  and valid_now=11024;
select user_id
from m.agency
where valid_now=3133
  and agency_id_id= '18368';
select agency_id
from m.agency
where agency_id_id= '6433'
  and valid_now=4330;
select user_id
from m.agency
where valid_now=18693
  and agency_id_id= '19904';
select COUNT(*)
from dv.notes_message
where user_id='6710'
  and agency_id_id= '6710'
  and notice_id= '6710'
  and route_id= '6710';
select COUNT(*)
from dv.notes_message
where user_id='9938'
  and agency_id_id= '9938'
  and notice_id= '9938'
  and route_id= '9938';
select agency_id
from m.agency
where agency_id_id= '2878'
  and valid_now=15100;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5832'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18587
  and agency_id_id= '9596';
select user_id
from m.agency
where valid_now=9368
  and agency_id_id= '4390';
select agency_id
from m.agency
where agency_id_id= '13779'
  and valid_now=96;
select agency_id
from m.agency
where agency_id_id= '5082'
  and valid_now=18704;
select user_id
from m.agency
where valid_now=2963
  and agency_id_id= '5627';
select user_id
from m.agency
where valid_now=3363
  and agency_id_id= '12978';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4858'
  and valid_now=4860;
select agency_id
from m.agency
where agency_id_id= '18799'
  and valid_now=4668;
select COUNT(*)
from dv.notes_message
where user_id='2166'
  and agency_id_id= '2166'
  and notice_id= '2166'
  and route_id= '2166';
select COUNT(*)
from dv.notes_message
where user_id='19669'
  and agency_id_id= '19669'
  and notice_id= '19669'
  and route_id= '19669';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13531'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12525'
  and agency_id_id= '12525'
  and notice_id= '12525'
  and route_id= '12525';
select COUNT(*)
from dv.notes_message
where user_id='10715'
  and agency_id_id= '10715'
  and notice_id= '10715'
  and route_id= '10715';
select COUNT(*)
from dv.notes_message
where user_id='5903'
  and agency_id_id= '5903'
  and notice_id= '5903'
  and route_id= '5903';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11936'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='234'
  and agency_id_id= '234'
  and notice_id= '234'
  and route_id= '234';
select agency_id
from m.agency
where agency_id_id= '7096'
  and valid_now=3095;
select COUNT(*)
from dv.notes_message
where user_id='3330'
  and agency_id_id= '3330'
  and notice_id= '3330'
  and route_id= '3330';
select COUNT(*)
from dv.notes_message
where user_id='2859'
  and agency_id_id= '2859'
  and notice_id= '2859'
  and route_id= '2859';
select COUNT(*)
from dv.notes_message
where user_id='19517'
  and agency_id_id= '19517'
  and notice_id= '19517'
  and route_id= '19517';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10075'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5774'
  and valid_now=17126;
select user_id
from m.agency
where valid_now=11006
  and agency_id_id= '15673';
select user_id
from m.agency
where valid_now=8075
  and agency_id_id= '7254';
select COUNT(*)
from dv.notes_message
where user_id='16853'
  and agency_id_id= '16853'
  and notice_id= '16853'
  and route_id= '16853';
select COUNT(*)
from dv.notes_message
where user_id='19851'
  and agency_id_id= '19851'
  and notice_id= '19851'
  and route_id= '19851';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16456'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15418'
  and valid_now=12666;
select agency_id
from m.agency
where agency_id_id= '2265'
  and valid_now=14200;
select user_id
from m.agency
where valid_now=3521
  and agency_id_id= '5356';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11908'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4905'
  and valid_now=2133;
select COUNT(*)
from dv.notes_message
where user_id='1137'
  and agency_id_id= '1137'
  and notice_id= '1137'
  and route_id= '1137';
select COUNT(*)
from dv.notes_message
where user_id='11382'
  and agency_id_id= '11382'
  and notice_id= '11382'
  and route_id= '11382';
select agency_id
from m.agency
where agency_id_id= '7341'
  and valid_now=3038;
select agency_id
from m.agency
where agency_id_id= '12076'
  and valid_now=12756;
select agency_id
from m.agency
where agency_id_id= '10086'
  and valid_now=802;
select agency_id
from m.agency
where agency_id_id= '5155'
  and valid_now=13990;
select agency_id
from m.agency
where agency_id_id= '9906'
  and valid_now=9219;
select user_id
from m.agency
where valid_now=1285
  and agency_id_id= '10103';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14233'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19838
  and agency_id_id= '1489';
select user_id
from m.agency
where valid_now=2927
  and agency_id_id= '4220';
select COUNT(*)
from dv.notes_message
where user_id='3111'
  and agency_id_id= '3111'
  and notice_id= '3111'
  and route_id= '3111';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13660'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19872'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7485
  and agency_id_id= '3659';
select agency_id
from m.agency
where agency_id_id= '15423'
  and valid_now=9082;
select agency_id
from m.agency
where agency_id_id= '6161'
  and valid_now=15532;
select user_id
from m.agency
where valid_now=13981
  and agency_id_id= '18709';
select COUNT(*)
from dv.notes_message
where user_id='531'
  and agency_id_id= '531'
  and notice_id= '531'
  and route_id= '531';
select agency_id
from m.agency
where agency_id_id= '6559'
  and valid_now=3709;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19043'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12723
  and agency_id_id= '7031';
select COUNT(*)
from dv.notes_message
where user_id='5604'
  and agency_id_id= '5604'
  and notice_id= '5604'
  and route_id= '5604';
select agency_id
from m.agency
where agency_id_id= '1523'
  and valid_now=10327;
select agency_id
from m.agency
where agency_id_id= '3898'
  and valid_now=2952;
select user_id
from m.agency
where valid_now=12425
  and agency_id_id= '19329';
select user_id
from m.agency
where valid_now=13214
  and agency_id_id= '9953';
select COUNT(*)
from dv.notes_message
where user_id='11661'
  and agency_id_id= '11661'
  and notice_id= '11661'
  and route_id= '11661';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18620'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2818'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19953
  and agency_id_id= '1642';
select user_id
from m.agency
where valid_now=3430
  and agency_id_id= '8353';
select COUNT(*)
from dv.notes_message
where user_id='2055'
  and agency_id_id= '2055'
  and notice_id= '2055'
  and route_id= '2055';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '87'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17337'
  and valid_now=11150;
select agency_id
from m.agency
where agency_id_id= '3161'
  and valid_now=19406;
select COUNT(*)
from dv.notes_message
where user_id='7643'
  and agency_id_id= '7643'
  and notice_id= '7643'
  and route_id= '7643';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7745'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14560'
  and valid_now=14582;
select agency_id
from m.agency
where agency_id_id= '7160'
  and valid_now=18178;
select COUNT(*)
from dv.notes_message
where user_id='8729'
  and agency_id_id= '8729'
  and notice_id= '8729'
  and route_id= '8729';
select a.agency_timezone
from m.agency a
where a.agency_id = '6064';
select COUNT(*)
from dv.notes_message
where user_id='13019'
  and agency_id_id= '13019'
  and notice_id= '13019'
  and route_id= '13019';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1393'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11852';
select COUNT(*)
from dv.notes_message
where user_id='2983'
  and agency_id_id= '2983'
  and notice_id= '2983'
  and route_id= '2983';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19736'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12991'
  and valid_now=14435;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6607'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4413'
  and valid_now=13605;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10541'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17772'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '7902';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1689'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18354'
  and valid_now=10436;
select COUNT(*)
from dv.notes_message
where user_id='5138'
  and agency_id_id= '5138'
  and notice_id= '5138'
  and route_id= '5138';
select COUNT(*)
from dv.notes_message
where user_id='13545'
  and agency_id_id= '13545'
  and notice_id= '13545'
  and route_id= '13545';
select a.agency_timezone
from m.agency a
where a.agency_id = '13550';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '178'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '915'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17269'
  and valid_now=9212;
select agency_id
from m.agency
where agency_id_id= '6349'
  and valid_now=19798;
select COUNT(*)
from dv.notes_message
where user_id='16261'
  and agency_id_id= '16261'
  and notice_id= '16261'
  and route_id= '16261';
select a.agency_timezone
from m.agency a
where a.agency_id = '15483';
select a.agency_timezone
from m.agency a
where a.agency_id = '11475';
select COUNT(*)
from dv.notes_message
where user_id='17'
  and agency_id_id= '17'
  and notice_id= '17'
  and route_id= '17';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3838'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3071
  and agency_id_id= '1748';
select a.agency_timezone
from m.agency a
where a.agency_id = '8861';
select a.agency_timezone
from m.agency a
where a.agency_id = '14500';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8862'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13261';
select a.agency_timezone
from m.agency a
where a.agency_id = '8625';
select a.agency_timezone
from m.agency a
where a.agency_id = '13117';
select agency_id
from m.agency
where agency_id_id= '19263'
  and valid_now=3133;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13373'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10321
  and agency_id_id= '6679';
select COUNT(*)
from dv.notes_message
where user_id='16105'
  and agency_id_id= '16105'
  and notice_id= '16105'
  and route_id= '16105';
select agency_id
from m.agency
where agency_id_id= '4870'
  and valid_now=12182;
select user_id
from m.agency
where valid_now=6198
  and agency_id_id= '19191';
select COUNT(*)
from dv.notes_message
where user_id='18259'
  and agency_id_id= '18259'
  and notice_id= '18259'
  and route_id= '18259';
select user_id
from m.agency
where valid_now=7951
  and agency_id_id= '12648';
select user_id
from m.agency
where valid_now=10269
  and agency_id_id= '743';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14857'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18005
  and agency_id_id= '12512';
select COUNT(*)
from dv.notes_message
where user_id='6310'
  and agency_id_id= '6310'
  and notice_id= '6310'
  and route_id= '6310';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17699'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12153
  and agency_id_id= '7340';
select user_id
from m.agency
where valid_now=7398
  and agency_id_id= '15931';
select COUNT(*)
from dv.notes_message
where user_id='1003'
  and agency_id_id= '1003'
  and notice_id= '1003'
  and route_id= '1003';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12577'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13966'
  and valid_now=10292;
select user_id
from m.agency
where valid_now=4131
  and agency_id_id= '3082';
select COUNT(*)
from dv.notes_message
where user_id='19487'
  and agency_id_id= '19487'
  and notice_id= '19487'
  and route_id= '19487';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11620'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7734'
  and valid_now=8665;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11965'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12229'
  and agency_id_id= '12229'
  and notice_id= '12229'
  and route_id= '12229';
select user_id
from m.agency
where valid_now=6491
  and agency_id_id= '19193';
select COUNT(*)
from dv.notes_message
where user_id='11274'
  and agency_id_id= '11274'
  and notice_id= '11274'
  and route_id= '11274';
select agency_id
from m.agency
where agency_id_id= '17276'
  and valid_now=8570;
select agency_id
from m.agency
where agency_id_id= '15801'
  and valid_now=9686;
select user_id
from m.agency
where valid_now=8877
  and agency_id_id= '14012';
select COUNT(*)
from dv.notes_message
where user_id='12404'
  and agency_id_id= '12404'
  and notice_id= '12404'
  and route_id= '12404';
select agency_id
from m.agency
where agency_id_id= '5217'
  and valid_now=9767;
select agency_id
from m.agency
where agency_id_id= '19628'
  and valid_now=11350;
select user_id
from m.agency
where valid_now=11496
  and agency_id_id= '9676';
select user_id
from m.agency
where valid_now=15379
  and agency_id_id= '8303';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1899'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1535'
  and agency_id_id= '1535'
  and notice_id= '1535'
  and route_id= '1535';
select user_id
from m.agency
where valid_now=8763
  and agency_id_id= '13829';
select COUNT(*)
from dv.notes_message
where user_id='16123'
  and agency_id_id= '16123'
  and notice_id= '16123'
  and route_id= '16123';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2787'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7342'
  and agency_id_id= '7342'
  and notice_id= '7342'
  and route_id= '7342';
select user_id
from m.agency
where valid_now=6322
  and agency_id_id= '1941';
select user_id
from m.agency
where valid_now=13776
  and agency_id_id= '4025';
select COUNT(*)
from dv.notes_message
where user_id='3024'
  and agency_id_id= '3024'
  and notice_id= '3024'
  and route_id= '3024';
select user_id
from m.agency
where valid_now=17775
  and agency_id_id= '10832';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8547'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16743'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2197
  and agency_id_id= '1603';
select agency_id
from m.agency
where agency_id_id= '15754'
  and valid_now=8981;
select user_id
from m.agency
where valid_now=14901
  and agency_id_id= '19799';
select COUNT(*)
from dv.notes_message
where user_id='13290'
  and agency_id_id= '13290'
  and notice_id= '13290'
  and route_id= '13290';
select agency_id
from m.agency
where agency_id_id= '11085'
  and valid_now=17728;
select agency_id
from m.agency
where agency_id_id= '16852'
  and valid_now=12057;
select user_id
from m.agency
where valid_now=17193
  and agency_id_id= '12785';
select user_id
from m.agency
where valid_now=13446
  and agency_id_id= '2974';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19559'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19714
  and agency_id_id= '977';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2624'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5120
  and agency_id_id= '2045';
select COUNT(*)
from dv.notes_message
where user_id='8887'
  and agency_id_id= '8887'
  and notice_id= '8887'
  and route_id= '8887';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15862'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12971'
  and agency_id_id= '12971'
  and notice_id= '12971'
  and route_id= '12971';
select COUNT(*)
from dv.notes_message
where user_id='1979'
  and agency_id_id= '1979'
  and notice_id= '1979'
  and route_id= '1979';
select agency_id
from m.agency
where agency_id_id= '1169'
  and valid_now=2743;
select COUNT(*)
from dv.notes_message
where user_id='19268'
  and agency_id_id= '19268'
  and notice_id= '19268'
  and route_id= '19268';
select user_id
from m.agency
where valid_now=15613
  and agency_id_id= '13630';
select COUNT(*)
from dv.notes_message
where user_id='17086'
  and agency_id_id= '17086'
  and notice_id= '17086'
  and route_id= '17086';
select COUNT(*)
from dv.notes_message
where user_id='17798'
  and agency_id_id= '17798'
  and notice_id= '17798'
  and route_id= '17798';
select agency_id
from m.agency
where agency_id_id= '9161'
  and valid_now=5680;
select COUNT(*)
from dv.notes_message
where user_id='7347'
  and agency_id_id= '7347'
  and notice_id= '7347'
  and route_id= '7347';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7706'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4362'
  and valid_now=11243;
select user_id
from m.agency
where valid_now=10100
  and agency_id_id= '2587';
select user_id
from m.agency
where valid_now=5351
  and agency_id_id= '9674';
select agency_id
from m.agency
where agency_id_id= '5045'
  and valid_now=10912;
select agency_id
from m.agency
where agency_id_id= '15317'
  and valid_now=12333;
select user_id
from m.agency
where valid_now=16531
  and agency_id_id= '7876';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17504'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4416'
  and valid_now=18228;
select user_id
from m.agency
where valid_now=13100
  and agency_id_id= '12347';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17090'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14848'
  and valid_now=17906;
select COUNT(*)
from dv.notes_message
where user_id='19939'
  and agency_id_id= '19939'
  and notice_id= '19939'
  and route_id= '19939';
select user_id
from m.agency
where valid_now=15480
  and agency_id_id= '4664';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10407'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13879'
  and valid_now=9737;
select user_id
from m.agency
where valid_now=8921
  and agency_id_id= '1275';
select COUNT(*)
from dv.notes_message
where user_id='4631'
  and agency_id_id= '4631'
  and notice_id= '4631'
  and route_id= '4631';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4255'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11405'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11057'
  and agency_id_id= '11057'
  and notice_id= '11057'
  and route_id= '11057';
select COUNT(*)
from dv.notes_message
where user_id='14082'
  and agency_id_id= '14082'
  and notice_id= '14082'
  and route_id= '14082';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2260'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15304
  and agency_id_id= '16673';
select COUNT(*)
from dv.notes_message
where user_id='14859'
  and agency_id_id= '14859'
  and notice_id= '14859'
  and route_id= '14859';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '52'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14849'
  and valid_now=11337;
select COUNT(*)
from dv.notes_message
where user_id='18038'
  and agency_id_id= '18038'
  and notice_id= '18038'
  and route_id= '18038';
select user_id
from m.agency
where valid_now=9972
  and agency_id_id= '4181';
select user_id
from m.agency
where valid_now=6153
  and agency_id_id= '8586';
select user_id
from m.agency
where valid_now=11519
  and agency_id_id= '10028';
select user_id
from m.agency
where valid_now=6341
  and agency_id_id= '7911';
select user_id
from m.agency
where valid_now=17975
  and agency_id_id= '16410';
select COUNT(*)
from dv.notes_message
where user_id='19071'
  and agency_id_id= '19071'
  and notice_id= '19071'
  and route_id= '19071';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7221'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9615'
  and valid_now=3887;
select COUNT(*)
from dv.notes_message
where user_id='6890'
  and agency_id_id= '6890'
  and notice_id= '6890'
  and route_id= '6890';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16093'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16676'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11169'
  and agency_id_id= '11169'
  and notice_id= '11169'
  and route_id= '11169';
select agency_id
from m.agency
where agency_id_id= '11445'
  and valid_now=582;
select COUNT(*)
from dv.notes_message
where user_id='3690'
  and agency_id_id= '3690'
  and notice_id= '3690'
  and route_id= '3690';
select agency_id
from m.agency
where agency_id_id= '11821'
  and valid_now=10506;
select COUNT(*)
from dv.notes_message
where user_id='12255'
  and agency_id_id= '12255'
  and notice_id= '12255'
  and route_id= '12255';
select COUNT(*)
from dv.notes_message
where user_id='9129'
  and agency_id_id= '9129'
  and notice_id= '9129'
  and route_id= '9129';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17795'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17250'
  and valid_now=5092;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12525'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6789'
  and valid_now=7638;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17665'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8108'
  and valid_now=5026;
select user_id
from m.agency
where valid_now=2107
  and agency_id_id= '7953';
select COUNT(*)
from dv.notes_message
where user_id='10001'
  and agency_id_id= '10001'
  and notice_id= '10001'
  and route_id= '10001';
select COUNT(*)
from dv.notes_message
where user_id='6872'
  and agency_id_id= '6872'
  and notice_id= '6872'
  and route_id= '6872';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15460'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8517'
  and agency_id_id= '8517'
  and notice_id= '8517'
  and route_id= '8517';
select COUNT(*)
from dv.notes_message
where user_id='1558'
  and agency_id_id= '1558'
  and notice_id= '1558'
  and route_id= '1558';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16333'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6051'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1438'
  and valid_now=12721;
select user_id
from m.agency
where valid_now=11546
  and agency_id_id= '14782';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18595'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '10530';
select COUNT(*)
from dv.notes_message
where user_id='14475'
  and agency_id_id= '14475'
  and notice_id= '14475'
  and route_id= '14475';
select COUNT(*)
from dv.notes_message
where user_id='8084'
  and agency_id_id= '8084'
  and notice_id= '8084'
  and route_id= '8084';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10555'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1414'
  and valid_now=5886;
select agency_id
from m.agency
where agency_id_id= '11454'
  and valid_now=18162;
select a.agency_timezone
from m.agency a
where a.agency_id = '5423';
select a.agency_timezone
from m.agency a
where a.agency_id = '1524';
select agency_id
from m.agency
where agency_id_id= '15433'
  and valid_now=1965;
select a.agency_timezone
from m.agency a
where a.agency_id = '9372';
select agency_id
from m.agency
where agency_id_id= '16123'
  and valid_now=3831;
select COUNT(*)
from dv.notes_message
where user_id='8324'
  and agency_id_id= '8324'
  and notice_id= '8324'
  and route_id= '8324';
select COUNT(*)
from dv.notes_message
where user_id='16339'
  and agency_id_id= '16339'
  and notice_id= '16339'
  and route_id= '16339';
select COUNT(*)
from dv.notes_message
where user_id='8286'
  and agency_id_id= '8286'
  and notice_id= '8286'
  and route_id= '8286';
select COUNT(*)
from dv.notes_message
where user_id='1909'
  and agency_id_id= '1909'
  and notice_id= '1909'
  and route_id= '1909';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10187'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5804';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5086'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13358'
  and agency_id_id= '13358'
  and notice_id= '13358'
  and route_id= '13358';
select COUNT(*)
from dv.notes_message
where user_id='19961'
  and agency_id_id= '19961'
  and notice_id= '19961'
  and route_id= '19961';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8096'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13624';
select agency_id
from m.agency
where agency_id_id= '98'
  and valid_now=8692;
select a.agency_timezone
from m.agency a
where a.agency_id = '3717';
select a.agency_timezone
from m.agency a
where a.agency_id = '16468';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8651'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '17742';
select a.agency_timezone
from m.agency a
where a.agency_id = '3719';
select agency_id
from m.agency
where agency_id_id= '19210'
  and valid_now=2426;
select COUNT(*)
from dv.notes_message
where user_id='15596'
  and agency_id_id= '15596'
  and notice_id= '15596'
  and route_id= '15596';
select user_id
from m.agency
where valid_now=7399
  and agency_id_id= '3980';
select user_id
from m.agency
where valid_now=11868
  and agency_id_id= '14697';
select user_id
from m.agency
where valid_now=5657
  and agency_id_id= '15815';
select agency_id
from m.agency
where agency_id_id= '4431'
  and valid_now=12222;
select agency_id
from m.agency
where agency_id_id= '14095'
  and valid_now=18462;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4583'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2133
  and agency_id_id= '12802';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5073'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10324
  and agency_id_id= '1587';
select user_id
from m.agency
where valid_now=17390
  and agency_id_id= '2546';
select COUNT(*)
from dv.notes_message
where user_id='18217'
  and agency_id_id= '18217'
  and notice_id= '18217'
  and route_id= '18217';
select COUNT(*)
from dv.notes_message
where user_id='9154'
  and agency_id_id= '9154'
  and notice_id= '9154'
  and route_id= '9154';
select COUNT(*)
from dv.notes_message
where user_id='13534'
  and agency_id_id= '13534'
  and notice_id= '13534'
  and route_id= '13534';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18902'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13245'
  and valid_now=9031;
select COUNT(*)
from dv.notes_message
where user_id='6674'
  and agency_id_id= '6674'
  and notice_id= '6674'
  and route_id= '6674';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17453'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18322'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19527
  and agency_id_id= '11870';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11375'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13438'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8881'
  and valid_now=2735;
select user_id
from m.agency
where valid_now=11748
  and agency_id_id= '5858';
select COUNT(*)
from dv.notes_message
where user_id='18320'
  and agency_id_id= '18320'
  and notice_id= '18320'
  and route_id= '18320';
select COUNT(*)
from dv.notes_message
where user_id='8612'
  and agency_id_id= '8612'
  and notice_id= '8612'
  and route_id= '8612';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8190'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17911'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13674'
  and valid_now=14886;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1422'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15961'
  and valid_now=2680;
select user_id
from m.agency
where valid_now=4718
  and agency_id_id= '119';
select COUNT(*)
from dv.notes_message
where user_id='13090'
  and agency_id_id= '13090'
  and notice_id= '13090'
  and route_id= '13090';
select COUNT(*)
from dv.notes_message
where user_id='13912'
  and agency_id_id= '13912'
  and notice_id= '13912'
  and route_id= '13912';
select agency_id
from m.agency
where agency_id_id= '4540'
  and valid_now=8196;
select agency_id
from m.agency
where agency_id_id= '14573'
  and valid_now=10662;
select user_id
from m.agency
where valid_now=18212
  and agency_id_id= '16210';
select COUNT(*)
from dv.notes_message
where user_id='5875'
  and agency_id_id= '5875'
  and notice_id= '5875'
  and route_id= '5875';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2816'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10394'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17442'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6933'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4563'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2549'
  and valid_now=17529;
select user_id
from m.agency
where valid_now=8299
  and agency_id_id= '8126';
select user_id
from m.agency
where valid_now=5326
  and agency_id_id= '12152';
select user_id
from m.agency
where valid_now=15916
  and agency_id_id= '16262';
select agency_id
from m.agency
where agency_id_id= '19620'
  and valid_now=16089;
select user_id
from m.agency
where valid_now=3095
  and agency_id_id= '11599';
select COUNT(*)
from dv.notes_message
where user_id='2316'
  and agency_id_id= '2316'
  and notice_id= '2316'
  and route_id= '2316';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15969'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18735'
  and valid_now=10018;
select COUNT(*)
from dv.notes_message
where user_id='8784'
  and agency_id_id= '8784'
  and notice_id= '8784'
  and route_id= '8784';
select agency_id
from m.agency
where agency_id_id= '7799'
  and valid_now=1032;
select COUNT(*)
from dv.notes_message
where user_id='14265'
  and agency_id_id= '14265'
  and notice_id= '14265'
  and route_id= '14265';
select COUNT(*)
from dv.notes_message
where user_id='15443'
  and agency_id_id= '15443'
  and notice_id= '15443'
  and route_id= '15443';
select COUNT(*)
from dv.notes_message
where user_id='15981'
  and agency_id_id= '15981'
  and notice_id= '15981'
  and route_id= '15981';
select agency_id
from m.agency
where agency_id_id= '8579'
  and valid_now=772;
select agency_id
from m.agency
where agency_id_id= '17461'
  and valid_now=15956;
select agency_id
from m.agency
where agency_id_id= '1092'
  and valid_now=9782;
select user_id
from m.agency
where valid_now=12156
  and agency_id_id= '10197';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3149'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7061'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11376'
  and valid_now=17372;
select user_id
from m.agency
where valid_now=12681
  and agency_id_id= '15714';
select user_id
from m.agency
where valid_now=15973
  and agency_id_id= '2503';
select COUNT(*)
from dv.notes_message
where user_id='5443'
  and agency_id_id= '5443'
  and notice_id= '5443'
  and route_id= '5443';
select agency_id
from m.agency
where agency_id_id= '4957'
  and valid_now=9382;
select COUNT(*)
from dv.notes_message
where user_id='9526'
  and agency_id_id= '9526'
  and notice_id= '9526'
  and route_id= '9526';
select COUNT(*)
from dv.notes_message
where user_id='830'
  and agency_id_id= '830'
  and notice_id= '830'
  and route_id= '830';
select agency_id
from m.agency
where agency_id_id= '4552'
  and valid_now=15239;
select agency_id
from m.agency
where agency_id_id= '11934'
  and valid_now=17544;
select user_id
from m.agency
where valid_now=10178
  and agency_id_id= '13770';
select user_id
from m.agency
where valid_now=14343
  and agency_id_id= '11190';
select agency_id
from m.agency
where agency_id_id= '11827'
  and valid_now=6585;
select COUNT(*)
from dv.notes_message
where user_id='13431'
  and agency_id_id= '13431'
  and notice_id= '13431'
  and route_id= '13431';
select COUNT(*)
from dv.notes_message
where user_id='6390'
  and agency_id_id= '6390'
  and notice_id= '6390'
  and route_id= '6390';
select COUNT(*)
from dv.notes_message
where user_id='4511'
  and agency_id_id= '4511'
  and notice_id= '4511'
  and route_id= '4511';
select COUNT(*)
from dv.notes_message
where user_id='4382'
  and agency_id_id= '4382'
  and notice_id= '4382'
  and route_id= '4382';
select a.agency_timezone
from m.agency a
where a.agency_id = '18143';
select a.agency_timezone
from m.agency a
where a.agency_id = '10894';
select user_id
from m.agency
where valid_now=17927
  and agency_id_id= '2135';
select COUNT(*)
from dv.notes_message
where user_id='7856'
  and agency_id_id= '7856'
  and notice_id= '7856'
  and route_id= '7856';
select COUNT(*)
from dv.notes_message
where user_id='18546'
  and agency_id_id= '18546'
  and notice_id= '18546'
  and route_id= '18546';
select COUNT(*)
from dv.notes_message
where user_id='4005'
  and agency_id_id= '4005'
  and notice_id= '4005'
  and route_id= '4005';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11659'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1955
  and agency_id_id= '14013';
select user_id
from m.agency
where valid_now=18223
  and agency_id_id= '12634';
select user_id
from m.agency
where valid_now=19974
  and agency_id_id= '5686';
select COUNT(*)
from dv.notes_message
where user_id='13835'
  and agency_id_id= '13835'
  and notice_id= '13835'
  and route_id= '13835';
select user_id
from m.agency
where valid_now=17208
  and agency_id_id= '267';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '458'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1412'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7582
  and agency_id_id= '4277';
select COUNT(*)
from dv.notes_message
where user_id='14314'
  and agency_id_id= '14314'
  and notice_id= '14314'
  and route_id= '14314';
select agency_id
from m.agency
where agency_id_id= '19754'
  and valid_now=221;
select agency_id
from m.agency
where agency_id_id= '9460'
  and valid_now=14874;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19266'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '14499'
  and valid_now=610;
select user_id
from m.agency
where valid_now=10054
  and agency_id_id= '16900';
select user_id
from m.agency
where valid_now=6667
  and agency_id_id= '2989';
select agency_id
from m.agency
where agency_id_id= '11973'
  and valid_now=9300;
select COUNT(*)
from dv.notes_message
where user_id='5214'
  and agency_id_id= '5214'
  and notice_id= '5214'
  and route_id= '5214';
select COUNT(*)
from dv.notes_message
where user_id='9644'
  and agency_id_id= '9644'
  and notice_id= '9644'
  and route_id= '9644';
select COUNT(*)
from dv.notes_message
where user_id='2040'
  and agency_id_id= '2040'
  and notice_id= '2040'
  and route_id= '2040';
select agency_id
from m.agency
where agency_id_id= '16279'
  and valid_now=19509;
select COUNT(*)
from dv.notes_message
where user_id='3224'
  and agency_id_id= '3224'
  and notice_id= '3224'
  and route_id= '3224';
select user_id
from m.agency
where valid_now=8762
  and agency_id_id= '19325';
select user_id
from m.agency
where valid_now=14487
  and agency_id_id= '19391';
select user_id
from m.agency
where valid_now=6913
  and agency_id_id= '10924';
select agency_id
from m.agency
where agency_id_id= '15232'
  and valid_now=17513;
select user_id
from m.agency
where valid_now=9104
  and agency_id_id= '14532';
select agency_id
from m.agency
where agency_id_id= '12607'
  and valid_now=12238;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7011'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8962'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17821'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19173
  and agency_id_id= '8415';
select COUNT(*)
from dv.notes_message
where user_id='598'
  and agency_id_id= '598'
  and notice_id= '598'
  and route_id= '598';
select COUNT(*)
from dv.notes_message
where user_id='4421'
  and agency_id_id= '4421'
  and notice_id= '4421'
  and route_id= '4421';
select agency_id
from m.agency
where agency_id_id= '783'
  and valid_now=5257;
select user_id
from m.agency
where valid_now=18780
  and agency_id_id= '8133';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19773'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3269'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7602'
  and valid_now=13956;
select COUNT(*)
from dv.notes_message
where user_id='13244'
  and agency_id_id= '13244'
  and notice_id= '13244'
  and route_id= '13244';
select user_id
from m.agency
where valid_now=6753
  and agency_id_id= '4058';
select COUNT(*)
from dv.notes_message
where user_id='19146'
  and agency_id_id= '19146'
  and notice_id= '19146'
  and route_id= '19146';
select agency_id
from m.agency
where agency_id_id= '10450'
  and valid_now=1917;
select user_id
from m.agency
where valid_now=6133
  and agency_id_id= '16295';
select COUNT(*)
from dv.notes_message
where user_id='2896'
  and agency_id_id= '2896'
  and notice_id= '2896'
  and route_id= '2896';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '330'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15769
  and agency_id_id= '12988';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8549'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7661'
  and valid_now=4334;
select user_id
from m.agency
where valid_now=5304
  and agency_id_id= '501';
select user_id
from m.agency
where valid_now=18683
  and agency_id_id= '3769';
select COUNT(*)
from dv.notes_message
where user_id='9339'
  and agency_id_id= '9339'
  and notice_id= '9339'
  and route_id= '9339';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14607'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5765'
  and valid_now=3545;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12585'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18449
  and agency_id_id= '6333';
select COUNT(*)
from dv.notes_message
where user_id='12755'
  and agency_id_id= '12755'
  and notice_id= '12755'
  and route_id= '12755';
select agency_id
from m.agency
where agency_id_id= '13859'
  and valid_now=16583;
select COUNT(*)
from dv.notes_message
where user_id='10390'
  and agency_id_id= '10390'
  and notice_id= '10390'
  and route_id= '10390';
select COUNT(*)
from dv.notes_message
where user_id='17137'
  and agency_id_id= '17137'
  and notice_id= '17137'
  and route_id= '17137';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19262'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4518'
  and valid_now=7228;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2644'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3652'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=17296
  and agency_id_id= '6033';
select a.agency_timezone
from m.agency a
where a.agency_id = '18595';
select user_id
from m.agency
where valid_now=12158
  and agency_id_id= '6249';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14150'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9162'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16033'
  and valid_now=14181;
select user_id
from m.agency
where valid_now=4996
  and agency_id_id= '10165';
select agency_id
from m.agency
where agency_id_id= '12371'
  and valid_now=4054;
select user_id
from m.agency
where valid_now=13593
  and agency_id_id= '17024';
select COUNT(*)
from dv.notes_message
where user_id='12653'
  and agency_id_id= '12653'
  and notice_id= '12653'
  and route_id= '12653';
select agency_id
from m.agency
where agency_id_id= '16725'
  and valid_now=15706;
select COUNT(*)
from dv.notes_message
where user_id='19863'
  and agency_id_id= '19863'
  and notice_id= '19863'
  and route_id= '19863';
select agency_id
from m.agency
where agency_id_id= '17046'
  and valid_now=16954;
select COUNT(*)
from dv.notes_message
where user_id='5459'
  and agency_id_id= '5459'
  and notice_id= '5459'
  and route_id= '5459';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_13100'
  and t.trip_id = 6286
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1057
  and agency_id_id= '19377';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_637'
  and t.trip_id = 17205
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18126'
  and valid_now=3786;
select COUNT(*)
from dv.notes_message
where user_id='18763'
  and agency_id_id= '18763'
  and notice_id= '18763'
  and route_id= '18763';
select agency_id
from m.agency
where agency_id_id= '2923'
  and valid_now=12415;
select COUNT(*)
from dv.notes_message
where user_id='3835'
  and agency_id_id= '3835'
  and notice_id= '3835'
  and route_id= '3835';
select COUNT(*)
from dv.notes_message
where user_id='18223'
  and agency_id_id= '18223'
  and notice_id= '18223'
  and route_id= '18223';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3131'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3639
  and agency_id_id= '8057';
select COUNT(*)
from dv.notes_message
where user_id='3554'
  and agency_id_id= '3554'
  and notice_id= '3554'
  and route_id= '3554';
select COUNT(*)
from dv.notes_message
where user_id='15093'
  and agency_id_id= '15093'
  and notice_id= '15093'
  and route_id= '15093';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_7105'
  and t.trip_id = 11177
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_14063'
  and t.trip_id = 4185
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_16467'
  and t.trip_id = 17767
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_5476'
  and t.trip_id = 5305
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1129
  and agency_id_id= '6556';
select COUNT(*)
from dv.notes_message
where user_id='12012'
  and agency_id_id= '12012'
  and notice_id= '12012'
  and route_id= '12012';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_503'
  and t.trip_id = 9666
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_18546'
  and t.trip_id = 17740
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4006'
  and agency_id_id= '4006'
  and notice_id= '4006'
  and route_id= '4006';
select agency_id
from m.agency
where agency_id_id= '13079'
  and valid_now=6834;
select user_id
from m.agency
where valid_now=1994
  and agency_id_id= '5237';
select user_id
from m.agency
where valid_now=2065
  and agency_id_id= '13731';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16746'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_11514'
  and t.trip_id = 10713
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_6593'
  and t.trip_id = 4374
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8051'
  and agency_id_id= '8051'
  and notice_id= '8051'
  and route_id= '8051';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '997'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11127'
  and agency_id_id= '11127'
  and notice_id= '11127'
  and route_id= '11127';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1103'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_1965'
  and t.trip_id = 163
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1037
  and agency_id_id= '15646';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_797'
  and t.trip_id = 15698
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_8913'
  and t.trip_id = 4940
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=17335
  and agency_id_id= '2418';
select a.agency_timezone
from m.agency a
where a.agency_id = '11862';
select COUNT(*)
from dv.notes_message
where user_id='6817'
  and agency_id_id= '6817'
  and notice_id= '6817'
  and route_id= '6817';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18284'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10686'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18754';
select a.agency_timezone
from m.agency a
where a.agency_id = '9013';
select user_id
from m.agency
where valid_now=6399
  and agency_id_id= '16035';
select COUNT(*)
from dv.notes_message
where user_id='3226'
  and agency_id_id= '3226'
  and notice_id= '3226'
  and route_id= '3226';
select COUNT(*)
from dv.notes_message
where user_id='12123'
  and agency_id_id= '12123'
  and notice_id= '12123'
  and route_id= '12123';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4547'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4795
  and agency_id_id= '9987';
select COUNT(*)
from dv.notes_message
where user_id='16888'
  and agency_id_id= '16888'
  and notice_id= '16888'
  and route_id= '16888';
select user_id
from m.agency
where valid_now=5858
  and agency_id_id= '4588';
select user_id
from m.agency
where valid_now=7353
  and agency_id_id= '15423';
select COUNT(*)
from dv.notes_message
where user_id='6376'
  and agency_id_id= '6376'
  and notice_id= '6376'
  and route_id= '6376';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_19061'
  and t.trip_id = 5255
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18340'
  and agency_id_id= '18340'
  and notice_id= '18340'
  and route_id= '18340';
select user_id
from m.agency
where valid_now=6079
  and agency_id_id= '1563';
select user_id
from m.agency
where valid_now=8986
  and agency_id_id= '7148';
select user_id
from m.agency
where valid_now=6300
  and agency_id_id= '5438';
select COUNT(*)
from dv.notes_message
where user_id='2319'
  and agency_id_id= '2319'
  and notice_id= '2319'
  and route_id= '2319';
select COUNT(*)
from dv.notes_message
where user_id='19787'
  and agency_id_id= '19787'
  and notice_id= '19787'
  and route_id= '19787';
select user_id
from m.agency
where valid_now=16269
  and agency_id_id= '16560';
select user_id
from m.agency
where valid_now=19819
  and agency_id_id= '11001';
select agency_id
from m.agency
where agency_id_id= '2494'
  and valid_now=16835;
select COUNT(*)
from dv.notes_message
where user_id='13957'
  and agency_id_id= '13957'
  and notice_id= '13957'
  and route_id= '13957';
select agency_id
from m.agency
where agency_id_id= '1534'
  and valid_now=17281;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7426'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1416'
  and valid_now=19344;
select agency_id
from m.agency
where agency_id_id= '12861'
  and valid_now=12457;
select user_id
from m.agency
where valid_now=2597
  and agency_id_id= '3103';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15695'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7167
  and agency_id_id= '18658';
select COUNT(*)
from dv.notes_message
where user_id='18863'
  and agency_id_id= '18863'
  and notice_id= '18863'
  and route_id= '18863';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17382'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4722
  and agency_id_id= '15665';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2596'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11164';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7490'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '16343';
select agency_id
from m.agency
where agency_id_id= '4117'
  and valid_now=8104;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19609'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '17382';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2737'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='10830'
  and agency_id_id= '10830'
  and notice_id= '10830'
  and route_id= '10830';
select COUNT(*)
from dv.notes_message
where user_id='4476'
  and agency_id_id= '4476'
  and notice_id= '4476'
  and route_id= '4476';
select COUNT(*)
from dv.notes_message
where user_id='11149'
  and agency_id_id= '11149'
  and notice_id= '11149'
  and route_id= '11149';
select user_id
from m.agency
where valid_now=10666
  and agency_id_id= '16613';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18684'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13841';
select COUNT(*)
from dv.notes_message
where user_id='8081'
  and agency_id_id= '8081'
  and notice_id= '8081'
  and route_id= '8081';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10141'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4613'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19233'
  and valid_now=2547;
select COUNT(*)
from dv.notes_message
where user_id='727'
  and agency_id_id= '727'
  and notice_id= '727'
  and route_id= '727';
select a.agency_timezone
from m.agency a
where a.agency_id = '4829';
select agency_id
from m.agency
where agency_id_id= '14289'
  and valid_now=7808;
select user_id
from m.agency
where valid_now=12285
  and agency_id_id= '2841';
select user_id
from m.agency
where valid_now=6149
  and agency_id_id= '8389';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11990'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15066
  and agency_id_id= '4709';
select COUNT(*)
from dv.notes_message
where user_id='16487'
  and agency_id_id= '16487'
  and notice_id= '16487'
  and route_id= '16487';
select agency_id
from m.agency
where agency_id_id= '693'
  and valid_now=1269;
select user_id
from m.agency
where valid_now=16083
  and agency_id_id= '13435';
select COUNT(*)
from dv.notes_message
where user_id='14940'
  and agency_id_id= '14940'
  and notice_id= '14940'
  and route_id= '14940';
select user_id
from m.agency
where valid_now=17981
  and agency_id_id= '913';
select COUNT(*)
from dv.notes_message
where user_id='18948'
  and agency_id_id= '18948'
  and notice_id= '18948'
  and route_id= '18948';
select user_id
from m.agency
where valid_now=1175
  and agency_id_id= '11611';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8525'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19514
  and agency_id_id= '9682';
select user_id
from m.agency
where valid_now=15784
  and agency_id_id= '17597';
select COUNT(*)
from dv.notes_message
where user_id='18696'
  and agency_id_id= '18696'
  and notice_id= '18696'
  and route_id= '18696';
select user_id
from m.agency
where valid_now=18991
  and agency_id_id= '10860';
select COUNT(*)
from dv.notes_message
where user_id='3297'
  and agency_id_id= '3297'
  and notice_id= '3297'
  and route_id= '3297';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4745'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10897
  and agency_id_id= '10140';
select user_id
from m.agency
where valid_now=4499
  and agency_id_id= '12046';
select COUNT(*)
from dv.notes_message
where user_id='11949'
  and agency_id_id= '11949'
  and notice_id= '11949'
  and route_id= '11949';
select COUNT(*)
from dv.notes_message
where user_id='12561'
  and agency_id_id= '12561'
  and notice_id= '12561'
  and route_id= '12561';
select agency_id
from m.agency
where agency_id_id= '19150'
  and valid_now=9345;
select COUNT(*)
from dv.notes_message
where user_id='13542'
  and agency_id_id= '13542'
  and notice_id= '13542'
  and route_id= '13542';
select agency_id
from m.agency
where agency_id_id= '14632'
  and valid_now=14152;
select agency_id
from m.agency
where agency_id_id= '3295'
  and valid_now=13244;
select agency_id
from m.agency
where agency_id_id= '16138'
  and valid_now=12950;
select agency_id
from m.agency
where agency_id_id= '1925'
  and valid_now=301;
select agency_id
from m.agency
where agency_id_id= '3162'
  and valid_now=595;
select user_id
from m.agency
where valid_now=1992
  and agency_id_id= '10703';
select COUNT(*)
from dv.notes_message
where user_id='2335'
  and agency_id_id= '2335'
  and notice_id= '2335'
  and route_id= '2335';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5587'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2825'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6001
  and agency_id_id= '1189';
select user_id
from m.agency
where valid_now=13095
  and agency_id_id= '5206';
select COUNT(*)
from dv.notes_message
where user_id='15665'
  and agency_id_id= '15665'
  and notice_id= '15665'
  and route_id= '15665';
select agency_id
from m.agency
where agency_id_id= '5128'
  and valid_now=12802;
select user_id
from m.agency
where valid_now=7541
  and agency_id_id= '10320';
select COUNT(*)
from dv.notes_message
where user_id='2761'
  and agency_id_id= '2761'
  and notice_id= '2761'
  and route_id= '2761';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15395'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7076'
  and agency_id_id= '7076'
  and notice_id= '7076'
  and route_id= '7076';
select user_id
from m.agency
where valid_now=6569
  and agency_id_id= '15871';
select COUNT(*)
from dv.notes_message
where user_id='13966'
  and agency_id_id= '13966'
  and notice_id= '13966'
  and route_id= '13966';
select COUNT(*)
from dv.notes_message
where user_id='19876'
  and agency_id_id= '19876'
  and notice_id= '19876'
  and route_id= '19876';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16529'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '16179'
  and valid_now=6659;
select user_id
from m.agency
where valid_now=9015
  and agency_id_id= '6475';
select user_id
from m.agency
where valid_now=1659
  and agency_id_id= '17524';
select COUNT(*)
from dv.notes_message
where user_id='13394'
  and agency_id_id= '13394'
  and notice_id= '13394'
  and route_id= '13394';
select agency_id
from m.agency
where agency_id_id= '9732'
  and valid_now=4777;
select agency_id
from m.agency
where agency_id_id= '16339'
  and valid_now=10281;
select agency_id
from m.agency
where agency_id_id= '15079'
  and valid_now=18403;
select user_id
from m.agency
where valid_now=13339
  and agency_id_id= '2754';
select user_id
from m.agency
where valid_now=18365
  and agency_id_id= '12834';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1487'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12643'
  and agency_id_id= '12643'
  and notice_id= '12643'
  and route_id= '12643';
select agency_id
from m.agency
where agency_id_id= '11994'
  and valid_now=15003;
select user_id
from m.agency
where valid_now=347
  and agency_id_id= '763';
select user_id
from m.agency
where valid_now=15902
  and agency_id_id= '3333';
select agency_id
from m.agency
where agency_id_id= '11379'
  and valid_now=11088;
select COUNT(*)
from dv.notes_message
where user_id='18895'
  and agency_id_id= '18895'
  and notice_id= '18895'
  and route_id= '18895';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8671'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16412'
  and agency_id_id= '16412'
  and notice_id= '16412'
  and route_id= '16412';
select agency_id
from m.agency
where agency_id_id= '10571'
  and valid_now=14715;
select user_id
from m.agency
where valid_now=3122
  and agency_id_id= '5706';
select COUNT(*)
from dv.notes_message
where user_id='19466'
  and agency_id_id= '19466'
  and notice_id= '19466'
  and route_id= '19466';
select user_id
from m.agency
where valid_now=6275
  and agency_id_id= '4819';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8836'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3584'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11539'
  and valid_now=7180;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10162'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1723'
  and valid_now=17580;
select agency_id
from m.agency
where agency_id_id= '11072'
  and valid_now=14578;
select COUNT(*)
from dv.notes_message
where user_id='5014'
  and agency_id_id= '5014'
  and notice_id= '5014'
  and route_id= '5014';
select user_id
from m.agency
where valid_now=14979
  and agency_id_id= '11227';
select COUNT(*)
from dv.notes_message
where user_id='3142'
  and agency_id_id= '3142'
  and notice_id= '3142'
  and route_id= '3142';
select COUNT(*)
from dv.notes_message
where user_id='19123'
  and agency_id_id= '19123'
  and notice_id= '19123'
  and route_id= '19123';
select user_id
from m.agency
where valid_now=16844
  and agency_id_id= '7588';
select user_id
from m.agency
where valid_now=32
  and agency_id_id= '6527';
select COUNT(*)
from dv.notes_message
where user_id='10566'
  and agency_id_id= '10566'
  and notice_id= '10566'
  and route_id= '10566';
select user_id
from m.agency
where valid_now=4281
  and agency_id_id= '18183';
select user_id
from m.agency
where valid_now=12946
  and agency_id_id= '10238';
select a.agency_timezone
from m.agency a
where a.agency_id = '19279';
select a.agency_timezone
from m.agency a
where a.agency_id = '10431';
select user_id
from m.agency
where valid_now=9811
  and agency_id_id= '14232';
select COUNT(*)
from dv.notes_message
where user_id='11518'
  and agency_id_id= '11518'
  and notice_id= '11518'
  and route_id= '11518';
select COUNT(*)
from dv.notes_message
where user_id='15885'
  and agency_id_id= '15885'
  and notice_id= '15885'
  and route_id= '15885';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1430'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18690'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '4678';
select COUNT(*)
from dv.notes_message
where user_id='17076'
  and agency_id_id= '17076'
  and notice_id= '17076'
  and route_id= '17076';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12741'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=55
  and agency_id_id= '14070';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18273'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19598'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3203
  and agency_id_id= '14748';
select COUNT(*)
from dv.notes_message
where user_id='13713'
  and agency_id_id= '13713'
  and notice_id= '13713'
  and route_id= '13713';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15358'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19402'
  and agency_id_id= '19402'
  and notice_id= '19402'
  and route_id= '19402';
select a.agency_timezone
from m.agency a
where a.agency_id = '18152';
select COUNT(*)
from dv.notes_message
where user_id='5777'
  and agency_id_id= '5777'
  and notice_id= '5777'
  and route_id= '5777';
select agency_id
from m.agency
where agency_id_id= '4756'
  and valid_now=12298;
select COUNT(*)
from dv.notes_message
where user_id='235'
  and agency_id_id= '235'
  and notice_id= '235'
  and route_id= '235';
select user_id
from m.agency
where valid_now=8692
  and agency_id_id= '2663';
select user_id
from m.agency
where valid_now=15792
  and agency_id_id= '8572';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9221'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13818'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1257'
  and agency_id_id= '1257'
  and notice_id= '1257'
  and route_id= '1257';
select user_id
from m.agency
where valid_now=2596
  and agency_id_id= '19423';
select user_id
from m.agency
where valid_now=19837
  and agency_id_id= '222';
select agency_id
from m.agency
where agency_id_id= '7540'
  and valid_now=11511;
select agency_id
from m.agency
where agency_id_id= '5365'
  and valid_now=8886;
select user_id
from m.agency
where valid_now=18958
  and agency_id_id= '298';
select user_id
from m.agency
where valid_now=1596
  and agency_id_id= '2794';
select COUNT(*)
from dv.notes_message
where user_id='7863'
  and agency_id_id= '7863'
  and notice_id= '7863'
  and route_id= '7863';
select agency_id
from m.agency
where agency_id_id= '8657'
  and valid_now=8304;
select agency_id
from m.agency
where agency_id_id= '10558'
  and valid_now=15148;
select COUNT(*)
from dv.notes_message
where user_id='9990'
  and agency_id_id= '9990'
  and notice_id= '9990'
  and route_id= '9990';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4185'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5725'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7117'
  and valid_now=17663;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8109'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7378'
  and agency_id_id= '7378'
  and notice_id= '7378'
  and route_id= '7378';
select agency_id
from m.agency
where agency_id_id= '5655'
  and valid_now=9734;
select COUNT(*)
from dv.notes_message
where user_id='274'
  and agency_id_id= '274'
  and notice_id= '274'
  and route_id= '274';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3589'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7611'
  and valid_now=7587;
select agency_id
from m.agency
where agency_id_id= '15621'
  and valid_now=19700;
select agency_id
from m.agency
where agency_id_id= '19254'
  and valid_now=6150;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8233'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9382'
  and valid_now=390;
select user_id
from m.agency
where valid_now=12546
  and agency_id_id= '16245';
select user_id
from m.agency
where valid_now=3720
  and agency_id_id= '814';
select user_id
from m.agency
where valid_now=17014
  and agency_id_id= '5934';
select COUNT(*)
from dv.notes_message
where user_id='18707'
  and agency_id_id= '18707'
  and notice_id= '18707'
  and route_id= '18707';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10689'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17320'
  and valid_now=18574;
select user_id
from m.agency
where valid_now=8575
  and agency_id_id= '6447';
select user_id
from m.agency
where valid_now=14277
  and agency_id_id= '19861';
select user_id
from m.agency
where valid_now=8998
  and agency_id_id= '3652';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13696'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8087'
  and valid_now=19865;
select COUNT(*)
from dv.notes_message
where user_id='15487'
  and agency_id_id= '15487'
  and notice_id= '15487'
  and route_id= '15487';
select COUNT(*)
from dv.notes_message
where user_id='12716'
  and agency_id_id= '12716'
  and notice_id= '12716'
  and route_id= '12716';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16861'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6745'
  and agency_id_id= '6745'
  and notice_id= '6745'
  and route_id= '6745';
select COUNT(*)
from dv.notes_message
where user_id='12927'
  and agency_id_id= '12927'
  and notice_id= '12927'
  and route_id= '12927';
select user_id
from m.agency
where valid_now=6051
  and agency_id_id= '4815';
select COUNT(*)
from dv.notes_message
where user_id='12827'
  and agency_id_id= '12827'
  and notice_id= '12827'
  and route_id= '12827';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13064'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11743'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5154
  and agency_id_id= '17359';
select COUNT(*)
from dv.notes_message
where user_id='14292'
  and agency_id_id= '14292'
  and notice_id= '14292'
  and route_id= '14292';
select COUNT(*)
from dv.notes_message
where user_id='19745'
  and agency_id_id= '19745'
  and notice_id= '19745'
  and route_id= '19745';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14064'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18361'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13668
  and agency_id_id= '11883';
select COUNT(*)
from dv.notes_message
where user_id='3552'
  and agency_id_id= '3552'
  and notice_id= '3552'
  and route_id= '3552';
select user_id
from m.agency
where valid_now=2559
  and agency_id_id= '952';
select user_id
from m.agency
where valid_now=3507
  and agency_id_id= '1240';
select agency_id
from m.agency
where agency_id_id= '7684'
  and valid_now=1828;
select agency_id
from m.agency
where agency_id_id= '14376'
  and valid_now=14486;
select user_id
from m.agency
where valid_now=1480
  and agency_id_id= '2253';
select COUNT(*)
from dv.notes_message
where user_id='14516'
  and agency_id_id= '14516'
  and notice_id= '14516'
  and route_id= '14516';
select COUNT(*)
from dv.notes_message
where user_id='2694'
  and agency_id_id= '2694'
  and notice_id= '2694'
  and route_id= '2694';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13293'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14101'
  and agency_id_id= '14101'
  and notice_id= '14101'
  and route_id= '14101';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6399'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7942'
  and agency_id_id= '7942'
  and notice_id= '7942'
  and route_id= '7942';
select agency_id
from m.agency
where agency_id_id= '15222'
  and valid_now=1449;
select agency_id
from m.agency
where agency_id_id= '2304'
  and valid_now=7123;
select user_id
from m.agency
where valid_now=1002
  and agency_id_id= '2864';
select user_id
from m.agency
where valid_now=19799
  and agency_id_id= '7092';
select user_id
from m.agency
where valid_now=17582
  and agency_id_id= '1739';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6926'
  and valid_now=9125;
select COUNT(*)
from dv.notes_message
where user_id='2167'
  and agency_id_id= '2167'
  and notice_id= '2167'
  and route_id= '2167';
select agency_id
from m.agency
where agency_id_id= '6215'
  and valid_now=13020;
select COUNT(*)
from dv.notes_message
where user_id='18992'
  and agency_id_id= '18992'
  and notice_id= '18992'
  and route_id= '18992';
select a.agency_timezone
from m.agency a
where a.agency_id = '4644';
select agency_id
from m.agency
where agency_id_id= '11796'
  and valid_now=15152;
select user_id
from m.agency
where valid_now=13709
  and agency_id_id= '11928';
select COUNT(*)
from dv.notes_message
where user_id='17800'
  and agency_id_id= '17800'
  and notice_id= '17800'
  and route_id= '17800';
select COUNT(*)
from dv.notes_message
where user_id='8236'
  and agency_id_id= '8236'
  and notice_id= '8236'
  and route_id= '8236';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1198'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11581'
  and valid_now=8996;
select COUNT(*)
from dv.notes_message
where user_id='12764'
  and agency_id_id= '12764'
  and notice_id= '12764'
  and route_id= '12764';
select a.agency_timezone
from m.agency a
where a.agency_id = '5483';
select a.agency_timezone
from m.agency a
where a.agency_id = '11482';
select user_id
from m.agency
where valid_now=1345
  and agency_id_id= '972';
select COUNT(*)
from dv.notes_message
where user_id='13228'
  and agency_id_id= '13228'
  and notice_id= '13228'
  and route_id= '13228';
select agency_id
from m.agency
where agency_id_id= '16440'
  and valid_now=92;
select user_id
from m.agency
where valid_now=19316
  and agency_id_id= '9689';
select COUNT(*)
from dv.notes_message
where user_id='17313'
  and agency_id_id= '17313'
  and notice_id= '17313'
  and route_id= '17313';
select a.agency_timezone
from m.agency a
where a.agency_id = '13456';
select agency_id
from m.agency
where agency_id_id= '16975'
  and valid_now=16280;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13166'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '6413';
select user_id
from m.agency
where valid_now=7111
  and agency_id_id= '13214';
select user_id
from m.agency
where valid_now=7819
  and agency_id_id= '10172';
select user_id
from m.agency
where valid_now=6858
  and agency_id_id= '4846';
select user_id
from m.agency
where valid_now=1855
  and agency_id_id= '1051';
select agency_id
from m.agency
where agency_id_id= '2433'
  and valid_now=12394;
select user_id
from m.agency
where valid_now=19616
  and agency_id_id= '18062';
select COUNT(*)
from dv.notes_message
where user_id='4436'
  and agency_id_id= '4436'
  and notice_id= '4436'
  and route_id= '4436';
select a.agency_timezone
from m.agency a
where a.agency_id = '17289';
select agency_id
from m.agency
where agency_id_id= '6280'
  and valid_now=16759;
select COUNT(*)
from dv.notes_message
where user_id='2535'
  and agency_id_id= '2535'
  and notice_id= '2535'
  and route_id= '2535';
select agency_id
from m.agency
where agency_id_id= '5633'
  and valid_now=1544;
select agency_id
from m.agency
where agency_id_id= '19741'
  and valid_now=9069;
select agency_id
from m.agency
where agency_id_id= '5640'
  and valid_now=18871;
select user_id
from m.agency
where valid_now=14317
  and agency_id_id= '15549';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17325'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7859
  and agency_id_id= '4344';
select user_id
from m.agency
where valid_now=19498
  and agency_id_id= '16702';
select user_id
from m.agency
where valid_now=14453
  and agency_id_id= '15487';
select user_id
from m.agency
where valid_now=2219
  and agency_id_id= '8332';
select user_id
from m.agency
where valid_now=2326
  and agency_id_id= '16395';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8249'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '12742';
select a.agency_timezone
from m.agency a
where a.agency_id = '11025';
select agency_id
from m.agency
where agency_id_id= '16210'
  and valid_now=9584;
select COUNT(*)
from dv.notes_message
where user_id='17676'
  and agency_id_id= '17676'
  and notice_id= '17676'
  and route_id= '17676';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2892'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6205'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3961
  and agency_id_id= '17507';
select user_id
from m.agency
where valid_now=13932
  and agency_id_id= '9669';
select agency_id
from m.agency
where agency_id_id= '14975'
  and valid_now=8970;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9859'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19542
  and agency_id_id= '11388';
select COUNT(*)
from dv.notes_message
where user_id='7720'
  and agency_id_id= '7720'
  and notice_id= '7720'
  and route_id= '7720';
select agency_id
from m.agency
where agency_id_id= '2503'
  and valid_now=3974;
select user_id
from m.agency
where valid_now=11973
  and agency_id_id= '11360';
select agency_id
from m.agency
where agency_id_id= '3592'
  and valid_now=14600;
select user_id
from m.agency
where valid_now=5791
  and agency_id_id= '9523';
select agency_id
from m.agency
where agency_id_id= '19648'
  and valid_now=5618;
select user_id
from m.agency
where valid_now=12772
  and agency_id_id= '15096';
select agency_id
from m.agency
where agency_id_id= '4108'
  and valid_now=8320;
select COUNT(*)
from dv.notes_message
where user_id='7380'
  and agency_id_id= '7380'
  and notice_id= '7380'
  and route_id= '7380';
select COUNT(*)
from dv.notes_message
where user_id='12571'
  and agency_id_id= '12571'
  and notice_id= '12571'
  and route_id= '12571';
select agency_id
from m.agency
where agency_id_id= '11493'
  and valid_now=14953;
select agency_id
from m.agency
where agency_id_id= '1918'
  and valid_now=16827;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1842'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9660'
  and agency_id_id= '9660'
  and notice_id= '9660'
  and route_id= '9660';
select user_id
from m.agency
where valid_now=878
  and agency_id_id= '13968';
select COUNT(*)
from dv.notes_message
where user_id='9527'
  and agency_id_id= '9527'
  and notice_id= '9527'
  and route_id= '9527';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5662'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2721
  and agency_id_id= '6313';
select COUNT(*)
from dv.notes_message
where user_id='11683'
  and agency_id_id= '11683'
  and notice_id= '11683'
  and route_id= '11683';
select COUNT(*)
from dv.notes_message
where user_id='799'
  and agency_id_id= '799'
  and notice_id= '799'
  and route_id= '799';
select COUNT(*)
from dv.notes_message
where user_id='12215'
  and agency_id_id= '12215'
  and notice_id= '12215'
  and route_id= '12215';
select COUNT(*)
from dv.notes_message
where user_id='13563'
  and agency_id_id= '13563'
  and notice_id= '13563'
  and route_id= '13563';
select agency_id
from m.agency
where agency_id_id= '7282'
  and valid_now=10356;
select agency_id
from m.agency
where agency_id_id= '15983'
  and valid_now=14762;
select agency_id
from m.agency
where agency_id_id= '5529'
  and valid_now=823;
select agency_id
from m.agency
where agency_id_id= '2371'
  and valid_now=6316;
select user_id
from m.agency
where valid_now=6766
  and agency_id_id= '1997';
select agency_id
from m.agency
where agency_id_id= '16044'
  and valid_now=15475;
select agency_id
from m.agency
where agency_id_id= '5090'
  and valid_now=14888;
select user_id
from m.agency
where valid_now=8442
  and agency_id_id= '572';
select user_id
from m.agency
where valid_now=16564
  and agency_id_id= '14132';
select a.agency_timezone
from m.agency a
where a.agency_id = '6748';
select user_id
from m.agency
where valid_now=18332
  and agency_id_id= '1428';
select user_id
from m.agency
where valid_now=1599
  and agency_id_id= '9606';
select a.agency_timezone
from m.agency a
where a.agency_id = '17904';
select a.agency_timezone
from m.agency a
where a.agency_id = '19057';
select COUNT(*)
from dv.notes_message
where user_id='4107'
  and agency_id_id= '4107'
  and notice_id= '4107'
  and route_id= '4107';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8291'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2150
  and agency_id_id= '14637';
select user_id
from m.agency
where valid_now=1828
  and agency_id_id= '5779';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10400'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7006'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '2952';
select COUNT(*)
from dv.notes_message
where user_id='12894'
  and agency_id_id= '12894'
  and notice_id= '12894'
  and route_id= '12894';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11733'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1368'
  and agency_id_id= '1368'
  and notice_id= '1368'
  and route_id= '1368';
select user_id
from m.agency
where valid_now=10949
  and agency_id_id= '11452';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10697'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6441
  and agency_id_id= '13418';
select COUNT(*)
from dv.notes_message
where user_id='4762'
  and agency_id_id= '4762'
  and notice_id= '4762'
  and route_id= '4762';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14913'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5550'
  and agency_id_id= '5550'
  and notice_id= '5550'
  and route_id= '5550';
select agency_id
from m.agency
where agency_id_id= '19342'
  and valid_now=14924;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19914'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11737
  and agency_id_id= '7675';
select user_id
from m.agency
where valid_now=13790
  and agency_id_id= '903';
select agency_id
from m.agency
where agency_id_id= '17946'
  and valid_now=13525;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4615'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13175
  and agency_id_id= '1094';
select user_id
from m.agency
where valid_now=3510
  and agency_id_id= '10411';
select agency_id
from m.agency
where agency_id_id= '2100'
  and valid_now=13468;
select agency_id
from m.agency
where agency_id_id= '5466'
  and valid_now=3949;
select user_id
from m.agency
where valid_now=1487
  and agency_id_id= '19647';
select COUNT(*)
from dv.notes_message
where user_id='7806'
  and agency_id_id= '7806'
  and notice_id= '7806'
  and route_id= '7806';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18051
  and agency_id_id= '15372';
select COUNT(*)
from dv.notes_message
where user_id='8482'
  and agency_id_id= '8482'
  and notice_id= '8482'
  and route_id= '8482';
select COUNT(*)
from dv.notes_message
where user_id='19565'
  and agency_id_id= '19565'
  and notice_id= '19565'
  and route_id= '19565';
select agency_id
from m.agency
where agency_id_id= '13439'
  and valid_now=2827;
select agency_id
from m.agency
where agency_id_id= '19031'
  and valid_now=11416;
select agency_id
from m.agency
where agency_id_id= '8152'
  and valid_now=18684;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10713'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6658
  and agency_id_id= '19790';
select user_id
from m.agency
where valid_now=18775
  and agency_id_id= '5412';
select agency_id
from m.agency
where agency_id_id= '17530'
  and valid_now=16627;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16673'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4141'
  and valid_now=619;
select agency_id
from m.agency
where agency_id_id= '14126'
  and valid_now=20;
select agency_id
from m.agency
where agency_id_id= '13003'
  and valid_now=3843;
select agency_id
from m.agency
where agency_id_id= '9744'
  and valid_now=8875;
select user_id
from m.agency
where valid_now=18934
  and agency_id_id= '10022';
select COUNT(*)
from dv.notes_message
where user_id='2016'
  and agency_id_id= '2016'
  and notice_id= '2016'
  and route_id= '2016';
select agency_id
from m.agency
where agency_id_id= '17687'
  and valid_now=1142;
select COUNT(*)
from dv.notes_message
where user_id='9590'
  and agency_id_id= '9590'
  and notice_id= '9590'
  and route_id= '9590';
select agency_id
from m.agency
where agency_id_id= '8760'
  and valid_now=17637;
select agency_id
from m.agency
where agency_id_id= '9470'
  and valid_now=13483;
select agency_id
from m.agency
where agency_id_id= '5836'
  and valid_now=10306;
select COUNT(*)
from dv.notes_message
where user_id='3440'
  and agency_id_id= '3440'
  and notice_id= '3440'
  and route_id= '3440';
select agency_id
from m.agency
where agency_id_id= '11178'
  and valid_now=16880;
select agency_id
from m.agency
where agency_id_id= '5729'
  and valid_now=6314;
select user_id
from m.agency
where valid_now=275
  and agency_id_id= '19606';
select agency_id
from m.agency
where agency_id_id= '11145'
  and valid_now=2634;
select agency_id
from m.agency
where agency_id_id= '680'
  and valid_now=7244;
select user_id
from m.agency
where valid_now=1867
  and agency_id_id= '17870';
select COUNT(*)
from dv.notes_message
where user_id='12431'
  and agency_id_id= '12431'
  and notice_id= '12431'
  and route_id= '12431';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6781'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19978
  and agency_id_id= '9036';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4429'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '615'
  and valid_now=3402;
select user_id
from m.agency
where valid_now=16938
  and agency_id_id= '8501';
select user_id
from m.agency
where valid_now=4497
  and agency_id_id= '14844';
select agency_id
from m.agency
where agency_id_id= '18688'
  and valid_now=18606;
select user_id
from m.agency
where valid_now=4373
  and agency_id_id= '17644';
select COUNT(*)
from dv.notes_message
where user_id='16565'
  and agency_id_id= '16565'
  and notice_id= '16565'
  and route_id= '16565';
select COUNT(*)
from dv.notes_message
where user_id='332'
  and agency_id_id= '332'
  and notice_id= '332'
  and route_id= '332';
select COUNT(*)
from dv.notes_message
where user_id='739'
  and agency_id_id= '739'
  and notice_id= '739'
  and route_id= '739';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14503'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16080
  and agency_id_id= '3086';
select agency_id
from m.agency
where agency_id_id= '17440'
  and valid_now=17199;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16041'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13754'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13140'
  and valid_now=17631;
select agency_id
from m.agency
where agency_id_id= '17539'
  and valid_now=7094;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17420'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13763'
  and agency_id_id= '13763'
  and notice_id= '13763'
  and route_id= '13763';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11206'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7010'
  and valid_now=11274;
select COUNT(*)
from dv.notes_message
where user_id='17097'
  and agency_id_id= '17097'
  and notice_id= '17097'
  and route_id= '17097';
select COUNT(*)
from dv.notes_message
where user_id='14541'
  and agency_id_id= '14541'
  and notice_id= '14541'
  and route_id= '14541';
select COUNT(*)
from dv.notes_message
where user_id='4809'
  and agency_id_id= '4809'
  and notice_id= '4809'
  and route_id= '4809';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2418'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='14761'
  and agency_id_id= '14761'
  and notice_id= '14761'
  and route_id= '14761';
select agency_id
from m.agency
where agency_id_id= '5169'
  and valid_now=14100;
select COUNT(*)
from dv.notes_message
where user_id='9596'
  and agency_id_id= '9596'
  and notice_id= '9596'
  and route_id= '9596';
select agency_id
from m.agency
where agency_id_id= '7863'
  and valid_now=1830;
select COUNT(*)
from dv.notes_message
where user_id='4098'
  and agency_id_id= '4098'
  and notice_id= '4098'
  and route_id= '4098';
select user_id
from m.agency
where valid_now=8997
  and agency_id_id= '13834';
select agency_id
from m.agency
where agency_id_id= '19515'
  and valid_now=2363;
select user_id
from m.agency
where valid_now=9309
  and agency_id_id= '12830';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8279'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12167
  and agency_id_id= '3419';
select COUNT(*)
from dv.notes_message
where user_id='12470'
  and agency_id_id= '12470'
  and notice_id= '12470'
  and route_id= '12470';
select agency_id
from m.agency
where agency_id_id= '5338'
  and valid_now=13783;
select user_id
from m.agency
where valid_now=11007
  and agency_id_id= '8437';
select COUNT(*)
from dv.notes_message
where user_id='5308'
  and agency_id_id= '5308'
  and notice_id= '5308'
  and route_id= '5308';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17828'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4116'
  and agency_id_id= '4116'
  and notice_id= '4116'
  and route_id= '4116';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13883'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12346'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13042'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='11185'
  and agency_id_id= '11185'
  and notice_id= '11185'
  and route_id= '11185';
select agency_id
from m.agency
where agency_id_id= '14545'
  and valid_now=5140;
select user_id
from m.agency
where valid_now=11188
  and agency_id_id= '14406';
select COUNT(*)
from dv.notes_message
where user_id='14570'
  and agency_id_id= '14570'
  and notice_id= '14570'
  and route_id= '14570';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7802'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12320'
  and agency_id_id= '12320'
  and notice_id= '12320'
  and route_id= '12320';
select COUNT(*)
from dv.notes_message
where user_id='3818'
  and agency_id_id= '3818'
  and notice_id= '3818'
  and route_id= '3818';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7237'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2779'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7666'
  and valid_now=18828;
select COUNT(*)
from dv.notes_message
where user_id='13479'
  and agency_id_id= '13479'
  and notice_id= '13479'
  and route_id= '13479';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1199'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19104'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2267
  and agency_id_id= '17369';
select agency_id
from m.agency
where agency_id_id= '10560'
  and valid_now=9915;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13293'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1108'
  and valid_now=10731;
select agency_id
from m.agency
where agency_id_id= '10129'
  and valid_now=10279;
select user_id
from m.agency
where valid_now=2368
  and agency_id_id= '10540';
select agency_id
from m.agency
where agency_id_id= '4441'
  and valid_now=12118;
select user_id
from m.agency
where valid_now=1523
  and agency_id_id= '455';
select COUNT(*)
from dv.notes_message
where user_id='9771'
  and agency_id_id= '9771'
  and notice_id= '9771'
  and route_id= '9771';
select agency_id
from m.agency
where agency_id_id= '19967'
  and valid_now=6625;
select user_id
from m.agency
where valid_now=14595
  and agency_id_id= '18597';
select COUNT(*)
from dv.notes_message
where user_id='5473'
  and agency_id_id= '5473'
  and notice_id= '5473'
  and route_id= '5473';
select COUNT(*)
from dv.notes_message
where user_id='10171'
  and agency_id_id= '10171'
  and notice_id= '10171'
  and route_id= '10171';
select COUNT(*)
from dv.notes_message
where user_id='4412'
  and agency_id_id= '4412'
  and notice_id= '4412'
  and route_id= '4412';
select COUNT(*)
from dv.notes_message
where user_id='18437'
  and agency_id_id= '18437'
  and notice_id= '18437'
  and route_id= '18437';
select agency_id
from m.agency
where agency_id_id= '19048'
  and valid_now=17342;
select agency_id
from m.agency
where agency_id_id= '7952'
  and valid_now=7688;
select user_id
from m.agency
where valid_now=9939
  and agency_id_id= '4569';
select user_id
from m.agency
where valid_now=2741
  and agency_id_id= '6246';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6415'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5812'
  and valid_now=7089;
select COUNT(*)
from dv.notes_message
where user_id='1995'
  and agency_id_id= '1995'
  and notice_id= '1995'
  and route_id= '1995';
select user_id
from m.agency
where valid_now=6210
  and agency_id_id= '6042';
select COUNT(*)
from dv.notes_message
where user_id='7647'
  and agency_id_id= '7647'
  and notice_id= '7647'
  and route_id= '7647';
select agency_id
from m.agency
where agency_id_id= '10334'
  and valid_now=3678;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19676'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16381
  and agency_id_id= '3781';
select user_id
from m.agency
where valid_now=18341
  and agency_id_id= '19408';
select user_id
from m.agency
where valid_now=9114
  and agency_id_id= '16264';
select COUNT(*)
from dv.notes_message
where user_id='13347'
  and agency_id_id= '13347'
  and notice_id= '13347'
  and route_id= '13347';
select a.agency_timezone
from m.agency a
where a.agency_id = '11399';
select user_id
from m.agency
where valid_now=12788
  and agency_id_id= '8543';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3063'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10526
  and agency_id_id= '19170';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17300'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '13700';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4200'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '18166';
select COUNT(*)
from dv.notes_message
where user_id='18468'
  and agency_id_id= '18468'
  and notice_id= '18468'
  and route_id= '18468';
select user_id
from m.agency
where valid_now=3774
  and agency_id_id= '726';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17182'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19235'
  and valid_now=15785;
select agency_id
from m.agency
where agency_id_id= '8491'
  and valid_now=9047;
select user_id
from m.agency
where valid_now=19571
  and agency_id_id= '874';
select user_id
from m.agency
where valid_now=8280
  and agency_id_id= '9532';
select COUNT(*)
from dv.notes_message
where user_id='15346'
  and agency_id_id= '15346'
  and notice_id= '15346'
  and route_id= '15346';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3772'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1213'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17804'
  and valid_now=3633;
select user_id
from m.agency
where valid_now=11901
  and agency_id_id= '12436';
select agency_id
from m.agency
where agency_id_id= '18253'
  and valid_now=1813;
select COUNT(*)
from dv.notes_message
where user_id='7539'
  and agency_id_id= '7539'
  and notice_id= '7539'
  and route_id= '7539';
select COUNT(*)
from dv.notes_message
where user_id='11103'
  and agency_id_id= '11103'
  and notice_id= '11103'
  and route_id= '11103';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4186'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '842'
  and valid_now=5668;
select COUNT(*)
from dv.notes_message
where user_id='1917'
  and agency_id_id= '1917'
  and notice_id= '1917'
  and route_id= '1917';
select user_id
from m.agency
where valid_now=4125
  and agency_id_id= '8487';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18075'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5540
  and agency_id_id= '18288';
select user_id
from m.agency
where valid_now=7
  and agency_id_id= '17993';
select COUNT(*)
from dv.notes_message
where user_id='4416'
  and agency_id_id= '4416'
  and notice_id= '4416'
  and route_id= '4416';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5843'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1138
  and agency_id_id= '16803';
select user_id
from m.agency
where valid_now=1268
  and agency_id_id= '5173';
select COUNT(*)
from dv.notes_message
where user_id='5296'
  and agency_id_id= '5296'
  and notice_id= '5296'
  and route_id= '5296';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5549'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8082'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9269'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14214
  and agency_id_id= '509';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11647'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5983
  and agency_id_id= '5927';
select agency_id
from m.agency
where agency_id_id= '19897'
  and valid_now=14728;
select user_id
from m.agency
where valid_now=19525
  and agency_id_id= '8135';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11651'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6472'
  and valid_now=2399;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3736'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12227
  and agency_id_id= '18871';
select COUNT(*)
from dv.notes_message
where user_id='3227'
  and agency_id_id= '3227'
  and notice_id= '3227'
  and route_id= '3227';
select COUNT(*)
from dv.notes_message
where user_id='1593'
  and agency_id_id= '1593'
  and notice_id= '1593'
  and route_id= '1593';
select COUNT(*)
from dv.notes_message
where user_id='6339'
  and agency_id_id= '6339'
  and notice_id= '6339'
  and route_id= '6339';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19792'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '11006';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13791'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='462'
  and agency_id_id= '462'
  and notice_id= '462'
  and route_id= '462';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19372'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18671'
  and valid_now=10987;
select agency_id
from m.agency
where agency_id_id= '15965'
  and valid_now=15303;
select user_id
from m.agency
where valid_now=17533
  and agency_id_id= '11410';
select user_id
from m.agency
where valid_now=5992
  and agency_id_id= '16707';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6437'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='10130'
  and agency_id_id= '10130'
  and notice_id= '10130'
  and route_id= '10130';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10722'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3789'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1165'
  and agency_id_id= '1165'
  and notice_id= '1165'
  and route_id= '1165';
select agency_id
from m.agency
where agency_id_id= '4370'
  and valid_now=3094;
select COUNT(*)
from dv.notes_message
where user_id='7793'
  and agency_id_id= '7793'
  and notice_id= '7793'
  and route_id= '7793';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17116'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=11241
  and agency_id_id= '17680';
select COUNT(*)
from dv.notes_message
where user_id='8216'
  and agency_id_id= '8216'
  and notice_id= '8216'
  and route_id= '8216';
select agency_id
from m.agency
where agency_id_id= '3304'
  and valid_now=2878;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10772'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8456'
  and valid_now=8489;
select user_id
from m.agency
where valid_now=17323
  and agency_id_id= '3233';
select user_id
from m.agency
where valid_now=17161
  and agency_id_id= '14147';
select COUNT(*)
from dv.notes_message
where user_id='16539'
  and agency_id_id= '16539'
  and notice_id= '16539'
  and route_id= '16539';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4911'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1260'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10285
  and agency_id_id= '18351';
select agency_id
from m.agency
where agency_id_id= '18532'
  and valid_now=11600;
select user_id
from m.agency
where valid_now=6679
  and agency_id_id= '9293';
select user_id
from m.agency
where valid_now=7646
  and agency_id_id= '9099';
select COUNT(*)
from dv.notes_message
where user_id='2266'
  and agency_id_id= '2266'
  and notice_id= '2266'
  and route_id= '2266';
select user_id
from m.agency
where valid_now=15151
  and agency_id_id= '15273';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4110'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7671
  and agency_id_id= '13076';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19117'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13640
  and agency_id_id= '15732';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17464'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6826'
  and agency_id_id= '6826'
  and notice_id= '6826'
  and route_id= '6826';
select user_id
from m.agency
where valid_now=17236
  and agency_id_id= '11632';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17316'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12880
  and agency_id_id= '4925';
select COUNT(*)
from dv.notes_message
where user_id='17770'
  and agency_id_id= '17770'
  and notice_id= '17770'
  and route_id= '17770';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '766'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8918'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15775'
  and agency_id_id= '15775'
  and notice_id= '15775'
  and route_id= '15775';
select user_id
from m.agency
where valid_now=19717
  and agency_id_id= '12746';
select user_id
from m.agency
where valid_now=16314
  and agency_id_id= '13475';
select COUNT(*)
from dv.notes_message
where user_id='5589'
  and agency_id_id= '5589'
  and notice_id= '5589'
  and route_id= '5589';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5461'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12553
  and agency_id_id= '1611';
select user_id
from m.agency
where valid_now=17433
  and agency_id_id= '6192';
select agency_id
from m.agency
where agency_id_id= '17165'
  and valid_now=13400;
select user_id
from m.agency
where valid_now=8203
  and agency_id_id= '4497';
select COUNT(*)
from dv.notes_message
where user_id='4387'
  and agency_id_id= '4387'
  and notice_id= '4387'
  and route_id= '4387';
select agency_id
from m.agency
where agency_id_id= '11250'
  and valid_now=7055;
select COUNT(*)
from dv.notes_message
where user_id='6991'
  and agency_id_id= '6991'
  and notice_id= '6991'
  and route_id= '6991';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11608'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15300
  and agency_id_id= '2828';
select user_id
from m.agency
where valid_now=17950
  and agency_id_id= '16237';
select COUNT(*)
from dv.notes_message
where user_id='10423'
  and agency_id_id= '10423'
  and notice_id= '10423'
  and route_id= '10423';
select COUNT(*)
from dv.notes_message
where user_id='8789'
  and agency_id_id= '8789'
  and notice_id= '8789'
  and route_id= '8789';
select COUNT(*)
from dv.notes_message
where user_id='17123'
  and agency_id_id= '17123'
  and notice_id= '17123'
  and route_id= '17123';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4986'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13058'
  and agency_id_id= '13058'
  and notice_id= '13058'
  and route_id= '13058';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13684'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5643
  and agency_id_id= '19402';
select agency_id
from m.agency
where agency_id_id= '2932'
  and valid_now=13464;
select user_id
from m.agency
where valid_now=17506
  and agency_id_id= '7213';
select agency_id
from m.agency
where agency_id_id= '14395'
  and valid_now=14041;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1536'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12069'
  and agency_id_id= '12069'
  and notice_id= '12069'
  and route_id= '12069';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5151'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13993'
  and agency_id_id= '13993'
  and notice_id= '13993'
  and route_id= '13993';
select COUNT(*)
from dv.notes_message
where user_id='8352'
  and agency_id_id= '8352'
  and notice_id= '8352'
  and route_id= '8352';
select agency_id
from m.agency
where agency_id_id= '14659'
  and valid_now=19381;
select agency_id
from m.agency
where agency_id_id= '10500'
  and valid_now=17634;
select user_id
from m.agency
where valid_now=6151
  and agency_id_id= '14706';
select agency_id
from m.agency
where agency_id_id= '5767'
  and valid_now=8630;
select user_id
from m.agency
where valid_now=10041
  and agency_id_id= '13572';
select user_id
from m.agency
where valid_now=17175
  and agency_id_id= '19826';
select agency_id
from m.agency
where agency_id_id= '11858'
  and valid_now=4306;
select user_id
from m.agency
where valid_now=6874
  and agency_id_id= '13613';
select agency_id
from m.agency
where agency_id_id= '10632'
  and valid_now=4288;
select COUNT(*)
from dv.notes_message
where user_id='2506'
  and agency_id_id= '2506'
  and notice_id= '2506'
  and route_id= '2506';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4360'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9158'
  and agency_id_id= '9158'
  and notice_id= '9158'
  and route_id= '9158';
select agency_id
from m.agency
where agency_id_id= '19219'
  and valid_now=8560;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12776'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4383'
  and valid_now=4626;
select agency_id
from m.agency
where agency_id_id= '9378'
  and valid_now=9768;
select COUNT(*)
from dv.notes_message
where user_id='720'
  and agency_id_id= '720'
  and notice_id= '720'
  and route_id= '720';
select user_id
from m.agency
where valid_now=1873
  and agency_id_id= '1486';
select user_id
from m.agency
where valid_now=17267
  and agency_id_id= '10076';
select COUNT(*)
from dv.notes_message
where user_id='18'
  and agency_id_id= '18'
  and notice_id= '18'
  and route_id= '18';
select user_id
from m.agency
where valid_now=9781
  and agency_id_id= '13382';
select agency_id
from m.agency
where agency_id_id= '2147'
  and valid_now=6059;
select user_id
from m.agency
where valid_now=964
  and agency_id_id= '14538';
select user_id
from m.agency
where valid_now=14379
  and agency_id_id= '17049';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11200'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3952
  and agency_id_id= '18011';
select COUNT(*)
from dv.notes_message
where user_id='16221'
  and agency_id_id= '16221'
  and notice_id= '16221'
  and route_id= '16221';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18441'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='18625'
  and agency_id_id= '18625'
  and notice_id= '18625'
  and route_id= '18625';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '652'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12683'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16851
  and agency_id_id= '11531';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18802'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1272'
  and valid_now=4383;
select user_id
from m.agency
where valid_now=3253
  and agency_id_id= '14741';
select user_id
from m.agency
where valid_now=7124
  and agency_id_id= '8627';
select agency_id
from m.agency
where agency_id_id= '15414'
  and valid_now=17846;
select user_id
from m.agency
where valid_now=3044
  and agency_id_id= '17403';
select user_id
from m.agency
where valid_now=3468
  and agency_id_id= '10970';
select user_id
from m.agency
where valid_now=1039
  and agency_id_id= '12225';
select a.agency_timezone
from m.agency a
where a.agency_id = '16062';
select COUNT(*)
from dv.notes_message
where user_id='4342'
  and agency_id_id= '4342'
  and notice_id= '4342'
  and route_id= '4342';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17550'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10853'
  and valid_now=12781;
select user_id
from m.agency
where valid_now=3678
  and agency_id_id= '3830';
select user_id
from m.agency
where valid_now=10748
  and agency_id_id= '10034';
select user_id
from m.agency
where valid_now=6739
  and agency_id_id= '18847';
select user_id
from m.agency
where valid_now=18644
  and agency_id_id= '16521';
select a.agency_timezone
from m.agency a
where a.agency_id = '5031';
select agency_id
from m.agency
where agency_id_id= '17390'
  and valid_now=17211;
select agency_id
from m.agency
where agency_id_id= '6532'
  and valid_now=9028;
select COUNT(*)
from dv.notes_message
where user_id='4407'
  and agency_id_id= '4407'
  and notice_id= '4407'
  and route_id= '4407';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5903'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13314'
  and valid_now=14190;
select user_id
from m.agency
where valid_now=10619
  and agency_id_id= '7905';
select agency_id
from m.agency
where agency_id_id= '15488'
  and valid_now=9151;
select agency_id
from m.agency
where agency_id_id= '15280'
  and valid_now=10323;
select user_id
from m.agency
where valid_now=7281
  and agency_id_id= '6399';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18431'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10532'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10804
  and agency_id_id= '4099';
select COUNT(*)
from dv.notes_message
where user_id='4168'
  and agency_id_id= '4168'
  and notice_id= '4168'
  and route_id= '4168';
select COUNT(*)
from dv.notes_message
where user_id='1081'
  and agency_id_id= '1081'
  and notice_id= '1081'
  and route_id= '1081';
select agency_id
from m.agency
where agency_id_id= '1490'
  and valid_now=11022;
select agency_id
from m.agency
where agency_id_id= '12655'
  and valid_now=4088;
select agency_id
from m.agency
where agency_id_id= '17610'
  and valid_now=18260;
select user_id
from m.agency
where valid_now=14368
  and agency_id_id= '14';
select user_id
from m.agency
where valid_now=5138
  and agency_id_id= '2016';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18898'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1508'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19784'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6983'
  and agency_id_id= '6983'
  and notice_id= '6983'
  and route_id= '6983';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10229'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17802'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12017'
  and valid_now=3163;
select agency_id
from m.agency
where agency_id_id= '1247'
  and valid_now=9275;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11156'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19448
  and agency_id_id= '2488';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19541'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1599'
  and agency_id_id= '1599'
  and notice_id= '1599'
  and route_id= '1599';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19864'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '15167'
  and valid_now=5469;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3350'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='1609'
  and agency_id_id= '1609'
  and notice_id= '1609'
  and route_id= '1609';
select COUNT(*)
from dv.notes_message
where user_id='151'
  and agency_id_id= '151'
  and notice_id= '151'
  and route_id= '151';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9283'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10975'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='13992'
  and agency_id_id= '13992'
  and notice_id= '13992'
  and route_id= '13992';
select agency_id
from m.agency
where agency_id_id= '9087'
  and valid_now=8647;
select COUNT(*)
from dv.notes_message
where user_id='2936'
  and agency_id_id= '2936'
  and notice_id= '2936'
  and route_id= '2936';
select agency_id
from m.agency
where agency_id_id= '4907'
  and valid_now=4660;
select agency_id
from m.agency
where agency_id_id= '1661'
  and valid_now=1293;
select COUNT(*)
from dv.notes_message
where user_id='4610'
  and agency_id_id= '4610'
  and notice_id= '4610'
  and route_id= '4610';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13984'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12572'
  and valid_now=4519;
select user_id
from m.agency
where valid_now=15591
  and agency_id_id= '19733';
select COUNT(*)
from dv.notes_message
where user_id='7199'
  and agency_id_id= '7199'
  and notice_id= '7199'
  and route_id= '7199';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11147'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12416
  and agency_id_id= '12372';
select user_id
from m.agency
where valid_now=10930
  and agency_id_id= '4231';
select COUNT(*)
from dv.notes_message
where user_id='15735'
  and agency_id_id= '15735'
  and notice_id= '15735'
  and route_id= '15735';
select user_id
from m.agency
where valid_now=6718
  and agency_id_id= '3765';
select COUNT(*)
from dv.notes_message
where user_id='11810'
  and agency_id_id= '11810'
  and notice_id= '11810'
  and route_id= '11810';
select agency_id
from m.agency
where agency_id_id= '14168'
  and valid_now=7827;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2584'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13130'
  and valid_now=12931;
select agency_id
from m.agency
where agency_id_id= '15488'
  and valid_now=14158;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7040'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8945'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10977'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6444
  and agency_id_id= '18803';
select COUNT(*)
from dv.notes_message
where user_id='14750'
  and agency_id_id= '14750'
  and notice_id= '14750'
  and route_id= '14750';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11870'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2679'
  and agency_id_id= '2679'
  and notice_id= '2679'
  and route_id= '2679';
select COUNT(*)
from dv.notes_message
where user_id='5218'
  and agency_id_id= '5218'
  and notice_id= '5218'
  and route_id= '5218';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17094'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9103'
  and valid_now=16910;
select COUNT(*)
from dv.notes_message
where user_id='2677'
  and agency_id_id= '2677'
  and notice_id= '2677'
  and route_id= '2677';
select user_id
from m.agency
where valid_now=18791
  and agency_id_id= '11682';
select agency_id
from m.agency
where agency_id_id= '6852'
  and valid_now=14288;
select COUNT(*)
from dv.notes_message
where user_id='11852'
  and agency_id_id= '11852'
  and notice_id= '11852'
  and route_id= '11852';
select a.agency_timezone
from m.agency a
where a.agency_id = '18306';
select a.agency_timezone
from m.agency a
where a.agency_id = '19752';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2121'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1659'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2791'
  and agency_id_id= '2791'
  and notice_id= '2791'
  and route_id= '2791';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1642'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3665'
  and valid_now=14605;
select COUNT(*)
from dv.notes_message
where user_id='1248'
  and agency_id_id= '1248'
  and notice_id= '1248'
  and route_id= '1248';
select agency_id
from m.agency
where agency_id_id= '16567'
  and valid_now=8480;
select a.agency_timezone
from m.agency a
where a.agency_id = '723';
select COUNT(*)
from dv.notes_message
where user_id='13320'
  and agency_id_id= '13320'
  and notice_id= '13320'
  and route_id= '13320';
select COUNT(*)
from dv.notes_message
where user_id='4425'
  and agency_id_id= '4425'
  and notice_id= '4425'
  and route_id= '4425';
select a.agency_timezone
from m.agency a
where a.agency_id = '2462';
select COUNT(*)
from dv.notes_message
where user_id='8256'
  and agency_id_id= '8256'
  and notice_id= '8256'
  and route_id= '8256';
select a.agency_timezone
from m.agency a
where a.agency_id = '18895';
select a.agency_timezone
from m.agency a
where a.agency_id = '18730';
select a.agency_timezone
from m.agency a
where a.agency_id = '14482';
select agency_id
from m.agency
where agency_id_id= '3427'
  and valid_now=6129;
select a.agency_timezone
from m.agency a
where a.agency_id = '6466';
select agency_id
from m.agency
where agency_id_id= '433'
  and valid_now=4539;
select agency_id
from m.agency
where agency_id_id= '14890'
  and valid_now=6736;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7326'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '5055';
select agency_id
from m.agency
where agency_id_id= '5753'
  and valid_now=7696;
select COUNT(*)
from dv.notes_message
where user_id='16623'
  and agency_id_id= '16623'
  and notice_id= '16623'
  and route_id= '16623';
select COUNT(*)
from dv.notes_message
where user_id='6823'
  and agency_id_id= '6823'
  and notice_id= '6823'
  and route_id= '6823';
select a.agency_timezone
from m.agency a
where a.agency_id = '16976';
select agency_id
from m.agency
where agency_id_id= '19899'
  and valid_now=8670;
select agency_id
from m.agency
where agency_id_id= '3346'
  and valid_now=15314;
select agency_id
from m.agency
where agency_id_id= '9687'
  and valid_now=15255;
select agency_id
from m.agency
where agency_id_id= '3550'
  and valid_now=11105;
select agency_id
from m.agency
where agency_id_id= '4250'
  and valid_now=1620;
select user_id
from m.agency
where valid_now=3362
  and agency_id_id= '12599';
select COUNT(*)
from dv.notes_message
where user_id='7947'
  and agency_id_id= '7947'
  and notice_id= '7947'
  and route_id= '7947';
select agency_id
from m.agency
where agency_id_id= '5948'
  and valid_now=8055;
select COUNT(*)
from dv.notes_message
where user_id='8319'
  and agency_id_id= '8319'
  and notice_id= '8319'
  and route_id= '8319';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2947'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9347
  and agency_id_id= '10810';
select user_id
from m.agency
where valid_now=9598
  and agency_id_id= '12665';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11418'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2188
  and agency_id_id= '6944';
select user_id
from m.agency
where valid_now=11588
  and agency_id_id= '14952';
select COUNT(*)
from dv.notes_message
where user_id='19029'
  and agency_id_id= '19029'
  and notice_id= '19029'
  and route_id= '19029';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6393'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16150'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1620'
  and valid_now=740;
select user_id
from m.agency
where valid_now=2464
  and agency_id_id= '11051';
select user_id
from m.agency
where valid_now=898
  and agency_id_id= '11711';
select agency_id
from m.agency
where agency_id_id= '2010'
  and valid_now=6411;
select user_id
from m.agency
where valid_now=10844
  and agency_id_id= '9995';
select user_id
from m.agency
where valid_now=847
  and agency_id_id= '4052';
select agency_id
from m.agency
where agency_id_id= '9727'
  and valid_now=11105;
select agency_id
from m.agency
where agency_id_id= '12143'
  and valid_now=11108;
select COUNT(*)
from dv.notes_message
where user_id='5990'
  and agency_id_id= '5990'
  and notice_id= '5990'
  and route_id= '5990';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11913'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '17372'
  and valid_now=17759;
select COUNT(*)
from dv.notes_message
where user_id='12056'
  and agency_id_id= '12056'
  and notice_id= '12056'
  and route_id= '12056';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3077'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9375
  and agency_id_id= '6857';
select user_id
from m.agency
where valid_now=14948
  and agency_id_id= '14738';
select agency_id
from m.agency
where agency_id_id= '3804'
  and valid_now=3567;
select agency_id
from m.agency
where agency_id_id= '1922'
  and valid_now=3299;
select agency_id
from m.agency
where agency_id_id= '8598'
  and valid_now=16362;
select user_id
from m.agency
where valid_now=3520
  and agency_id_id= '8826';
select COUNT(*)
from dv.notes_message
where user_id='2752'
  and agency_id_id= '2752'
  and notice_id= '2752'
  and route_id= '2752';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17015'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=2092
  and agency_id_id= '18422';
select COUNT(*)
from dv.notes_message
where user_id='17370'
  and agency_id_id= '17370'
  and notice_id= '17370'
  and route_id= '17370';
select user_id
from m.agency
where valid_now=14779
  and agency_id_id= '1651';
select COUNT(*)
from dv.notes_message
where user_id='17889'
  and agency_id_id= '17889'
  and notice_id= '17889'
  and route_id= '17889';
select agency_id
from m.agency
where agency_id_id= '15084'
  and valid_now=3411;
select user_id
from m.agency
where valid_now=5028
  and agency_id_id= '19724';
select agency_id
from m.agency
where agency_id_id= '6531'
  and valid_now=7996;
select agency_id
from m.agency
where agency_id_id= '17214'
  and valid_now=14371;
select agency_id
from m.agency
where agency_id_id= '19424'
  and valid_now=13673;
select COUNT(*)
from dv.notes_message
where user_id='6469'
  and agency_id_id= '6469'
  and notice_id= '6469'
  and route_id= '6469';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18472'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18619'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14294
  and agency_id_id= '1184';
select agency_id
from m.agency
where agency_id_id= '6971'
  and valid_now=9733;
select COUNT(*)
from dv.notes_message
where user_id='2882'
  and agency_id_id= '2882'
  and notice_id= '2882'
  and route_id= '2882';
select agency_id
from m.agency
where agency_id_id= '5382'
  and valid_now=3671;
select agency_id
from m.agency
where agency_id_id= '16779'
  and valid_now=3057;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18576'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5050'
  and agency_id_id= '5050'
  and notice_id= '5050'
  and route_id= '5050';
select a.agency_timezone
from m.agency a
where a.agency_id = '15595';
select user_id
from m.agency
where valid_now=14281
  and agency_id_id= '4314';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7776'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8997
  and agency_id_id= '2969';
select COUNT(*)
from dv.notes_message
where user_id='18512'
  and agency_id_id= '18512'
  and notice_id= '18512'
  and route_id= '18512';
select COUNT(*)
from dv.notes_message
where user_id='14773'
  and agency_id_id= '14773'
  and notice_id= '14773'
  and route_id= '14773';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11941'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '9818';
select a.agency_timezone
from m.agency a
where a.agency_id = '14020';
select COUNT(*)
from dv.notes_message
where user_id='3397'
  and agency_id_id= '3397'
  and notice_id= '3397'
  and route_id= '3397';
select user_id
from m.agency
where valid_now=2002
  and agency_id_id= '8833';
select COUNT(*)
from dv.notes_message
where user_id='11371'
  and agency_id_id= '11371'
  and notice_id= '11371'
  and route_id= '11371';
select COUNT(*)
from dv.notes_message
where user_id='16469'
  and agency_id_id= '16469'
  and notice_id= '16469'
  and route_id= '16469';
select user_id
from m.agency
where valid_now=9382
  and agency_id_id= '15699';
select user_id
from m.agency
where valid_now=15220
  and agency_id_id= '1644';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10927'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '4651'
  and valid_now=17664;
select user_id
from m.agency
where valid_now=11158
  and agency_id_id= '6594';
select COUNT(*)
from dv.notes_message
where user_id='15901'
  and agency_id_id= '15901'
  and notice_id= '15901'
  and route_id= '15901';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10929'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14193
  and agency_id_id= '5840';
select COUNT(*)
from dv.notes_message
where user_id='3991'
  and agency_id_id= '3991'
  and notice_id= '3991'
  and route_id= '3991';
select agency_id
from m.agency
where agency_id_id= '2712'
  and valid_now=2499;
select agency_id
from m.agency
where agency_id_id= '3196'
  and valid_now=13743;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16824'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16267'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2757'
  and valid_now=2529;
select agency_id
from m.agency
where agency_id_id= '6723'
  and valid_now=6817;
select user_id
from m.agency
where valid_now=18448
  and agency_id_id= '18515';
select user_id
from m.agency
where valid_now=16437
  and agency_id_id= '8707';
select COUNT(*)
from dv.notes_message
where user_id='1018'
  and agency_id_id= '1018'
  and notice_id= '1018'
  and route_id= '1018';
select user_id
from m.agency
where valid_now=466
  and agency_id_id= '13805';
select COUNT(*)
from dv.notes_message
where user_id='8408'
  and agency_id_id= '8408'
  and notice_id= '8408'
  and route_id= '8408';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2497'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8081'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11811'
  and valid_now=14351;
select COUNT(*)
from dv.notes_message
where user_id='6051'
  and agency_id_id= '6051'
  and notice_id= '6051'
  and route_id= '6051';
select COUNT(*)
from dv.notes_message
where user_id='47'
  and agency_id_id= '47'
  and notice_id= '47'
  and route_id= '47';
select agency_id
from m.agency
where agency_id_id= '19509'
  and valid_now=2840;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1675'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9634'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16829'
  and agency_id_id= '16829'
  and notice_id= '16829'
  and route_id= '16829';
select COUNT(*)
from dv.notes_message
where user_id='14630'
  and agency_id_id= '14630'
  and notice_id= '14630'
  and route_id= '14630';
select agency_id
from m.agency
where agency_id_id= '15683'
  and valid_now=18381;
select COUNT(*)
from dv.notes_message
where user_id='14429'
  and agency_id_id= '14429'
  and notice_id= '14429'
  and route_id= '14429';
select agency_id
from m.agency
where agency_id_id= '15958'
  and valid_now=16643;
select user_id
from m.agency
where valid_now=6973
  and agency_id_id= '473';
select COUNT(*)
from dv.notes_message
where user_id='18114'
  and agency_id_id= '18114'
  and notice_id= '18114'
  and route_id= '18114';
select agency_id
from m.agency
where agency_id_id= '14949'
  and valid_now=13513;
select agency_id
from m.agency
where agency_id_id= '15428'
  and valid_now=14783;
select user_id
from m.agency
where valid_now=19200
  and agency_id_id= '2383';
select agency_id
from m.agency
where agency_id_id= '9259'
  and valid_now=19013;
select agency_id
from m.agency
where agency_id_id= '3037'
  and valid_now=11119;
select COUNT(*)
from dv.notes_message
where user_id='4639'
  and agency_id_id= '4639'
  and notice_id= '4639'
  and route_id= '4639';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9122'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1924'
  and valid_now=18459;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17928'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '6351'
  and valid_now=3445;
select agency_id
from m.agency
where agency_id_id= '14401'
  and valid_now=16448;
select COUNT(*)
from dv.notes_message
where user_id='9673'
  and agency_id_id= '9673'
  and notice_id= '9673'
  and route_id= '9673';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2691'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17693'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7593'
  and agency_id_id= '7593'
  and notice_id= '7593'
  and route_id= '7593';
select agency_id
from m.agency
where agency_id_id= '13275'
  and valid_now=13514;
select COUNT(*)
from dv.notes_message
where user_id='7538'
  and agency_id_id= '7538'
  and notice_id= '7538'
  and route_id= '7538';
select user_id
from m.agency
where valid_now=3898
  and agency_id_id= '4536';
select user_id
from m.agency
where valid_now=8441
  and agency_id_id= '6589';
select agency_id
from m.agency
where agency_id_id= '12039'
  and valid_now=19593;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6841'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16309'
  and agency_id_id= '16309'
  and notice_id= '16309'
  and route_id= '16309';
select COUNT(*)
from dv.notes_message
where user_id='11179'
  and agency_id_id= '11179'
  and notice_id= '11179'
  and route_id= '11179';
select COUNT(*)
from dv.notes_message
where user_id='19483'
  and agency_id_id= '19483'
  and notice_id= '19483'
  and route_id= '19483';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1797'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7012'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14835'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17952'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6643
  and agency_id_id= '3168';
select user_id
from m.agency
where valid_now=8495
  and agency_id_id= '13023';
select COUNT(*)
from dv.notes_message
where user_id='15475'
  and agency_id_id= '15475'
  and notice_id= '15475'
  and route_id= '15475';
select COUNT(*)
from dv.notes_message
where user_id='16808'
  and agency_id_id= '16808'
  and notice_id= '16808'
  and route_id= '16808';
select user_id
from m.agency
where valid_now=53
  and agency_id_id= '795';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17740'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19679'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14559'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12276
  and agency_id_id= '9202';
select user_id
from m.agency
where valid_now=686
  and agency_id_id= '537';
select COUNT(*)
from dv.notes_message
where user_id='9172'
  and agency_id_id= '9172'
  and notice_id= '9172'
  and route_id= '9172';
select user_id
from m.agency
where valid_now=17866
  and agency_id_id= '14210';
select COUNT(*)
from dv.notes_message
where user_id='15207'
  and agency_id_id= '15207'
  and notice_id= '15207'
  and route_id= '15207';
select agency_id
from m.agency
where agency_id_id= '14963'
  and valid_now=19328;
select user_id
from m.agency
where valid_now=19541
  and agency_id_id= '13299';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12626'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=9844
  and agency_id_id= '1895';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4103'
  and agency_id_id= '4103'
  and notice_id= '4103'
  and route_id= '4103';
select user_id
from m.agency
where valid_now=16815
  and agency_id_id= '16520';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15738'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18123
  and agency_id_id= '877';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5378'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '9226';
select user_id
from m.agency
where valid_now=12139
  and agency_id_id= '15765';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5367'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='4217'
  and agency_id_id= '4217'
  and notice_id= '4217'
  and route_id= '4217';
select COUNT(*)
from dv.notes_message
where user_id='15047'
  and agency_id_id= '15047'
  and notice_id= '15047'
  and route_id= '15047';
select user_id
from m.agency
where valid_now=857
  and agency_id_id= '7184';
select a.agency_timezone
from m.agency a
where a.agency_id = '4087';
select a.agency_timezone
from m.agency a
where a.agency_id = '10842';
select user_id
from m.agency
where valid_now=16999
  and agency_id_id= '16710';
select user_id
from m.agency
where valid_now=1131
  and agency_id_id= '7461';
select COUNT(*)
from dv.notes_message
where user_id='17715'
  and agency_id_id= '17715'
  and notice_id= '17715'
  and route_id= '17715';
select COUNT(*)
from dv.notes_message
where user_id='17349'
  and agency_id_id= '17349'
  and notice_id= '17349'
  and route_id= '17349';
select COUNT(*)
from dv.notes_message
where user_id='9836'
  and agency_id_id= '9836'
  and notice_id= '9836'
  and route_id= '9836';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6460'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '8011';
select COUNT(*)
from dv.notes_message
where user_id='13348'
  and agency_id_id= '13348'
  and notice_id= '13348'
  and route_id= '13348';
select COUNT(*)
from dv.notes_message
where user_id='17476'
  and agency_id_id= '17476'
  and notice_id= '17476'
  and route_id= '17476';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3729'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='8758'
  and agency_id_id= '8758'
  and notice_id= '8758'
  and route_id= '8758';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8922'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11565'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12241
  and agency_id_id= '210';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4830'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '7891'
  and valid_now=12819;
select agency_id
from m.agency
where agency_id_id= '10644'
  and valid_now=10673;
select agency_id
from m.agency
where agency_id_id= '3118'
  and valid_now=12000;
select user_id
from m.agency
where valid_now=6369
  and agency_id_id= '3260';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11914'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=4721
  and agency_id_id= '11067';
select a.agency_timezone
from m.agency a
where a.agency_id = '1910';
select user_id
from m.agency
where valid_now=18578
  and agency_id_id= '17124';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3851'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3120'
  and agency_id_id= '3120'
  and notice_id= '3120'
  and route_id= '3120';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9579'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '7461';
select COUNT(*)
from dv.notes_message
where user_id='15767'
  and agency_id_id= '15767'
  and notice_id= '15767'
  and route_id= '15767';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5859'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10415
  and agency_id_id= '1717';
select COUNT(*)
from dv.notes_message
where user_id='9058'
  and agency_id_id= '9058'
  and notice_id= '9058'
  and route_id= '9058';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19894'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1370'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7731'
  and agency_id_id= '7731'
  and notice_id= '7731'
  and route_id= '7731';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11142'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '12621';
select user_id
from m.agency
where valid_now=11565
  and agency_id_id= '7629';
select COUNT(*)
from dv.notes_message
where user_id='4594'
  and agency_id_id= '4594'
  and notice_id= '4594'
  and route_id= '4594';
select a.agency_timezone
from m.agency a
where a.agency_id = '11783';
select user_id
from m.agency
where valid_now=820
  and agency_id_id= '16061';
select a.agency_timezone
from m.agency a
where a.agency_id = '16721';
select a.agency_timezone
from m.agency a
where a.agency_id = '17331';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3224'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '15784';
select COUNT(*)
from dv.notes_message
where user_id='7964'
  and agency_id_id= '7964'
  and notice_id= '7964'
  and route_id= '7964';
select a.agency_timezone
from m.agency a
where a.agency_id = '16153';
select a.agency_timezone
from m.agency a
where a.agency_id = '252';
select COUNT(*)
from dv.notes_message
where user_id='10425'
  and agency_id_id= '10425'
  and notice_id= '10425'
  and route_id= '10425';
select COUNT(*)
from dv.notes_message
where user_id='9359'
  and agency_id_id= '9359'
  and notice_id= '9359'
  and route_id= '9359';
select a.agency_timezone
from m.agency a
where a.agency_id = '13713';
select COUNT(*)
from dv.notes_message
where user_id='16734'
  and agency_id_id= '16734'
  and notice_id= '16734'
  and route_id= '16734';
select agency_id
from m.agency
where agency_id_id= '6865'
  and valid_now=921;
select user_id
from m.agency
where valid_now=15483
  and agency_id_id= '162';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_12754'
  and t.trip_id = 14042
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select a.agency_timezone
from m.agency a
where a.agency_id = '4286';
select agency_id
from m.agency
where agency_id_id= '14458'
  and valid_now=6559;
select agency_id
from m.agency
where agency_id_id= '1923'
  and valid_now=16612;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3335'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=223
  and agency_id_id= '10043';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17464'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4758'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9665'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7609'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_7803'
  and t.trip_id = 119
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12813
  and agency_id_id= '4985';
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_12803'
  and t.trip_id = 490
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13960'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_3208'
  and t.trip_id = 11904
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select distinct a.agency_id
from m.agency a,
     m.calendar c,
     m.trip t
where c.agency_id = a.agency_id
  and t.agency_id = a.agency_id
  and a.avl_agency_name = 'dummy_avl_agency_name_17005'
  and t.trip_id = 17182
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date+1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12153'
  and valid_now=10844;
select agency_id
from m.agency
where agency_id_id= '6012'
  and valid_now=16613;
select agency_id
from m.agency
where agency_id_id= '777'
  and valid_now=11887;
select a.agency_timezone
from m.agency a
where a.agency_id = '4515';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9447'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5084'
  and valid_now=10688;
select COUNT(*)
from dv.notes_message
where user_id='3509'
  and agency_id_id= '3509'
  and notice_id= '3509'
  and route_id= '3509';
select agency_id
from m.agency
where agency_id_id= '2156'
  and valid_now=6757;
select agency_id
from m.agency
where agency_id_id= '6410'
  and valid_now=11952;
select COUNT(*)
from dv.notes_message
where user_id='10975'
  and agency_id_id= '10975'
  and notice_id= '10975'
  and route_id= '10975';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2495'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='9281'
  and agency_id_id= '9281'
  and notice_id= '9281'
  and route_id= '9281';
select COUNT(*)
from dv.notes_message
where user_id='5838'
  and agency_id_id= '5838'
  and notice_id= '5838'
  and route_id= '5838';
select agency_id
from m.agency
where agency_id_id= '5016'
  and valid_now=19832;
select COUNT(*)
from dv.notes_message
where user_id='1928'
  and agency_id_id= '1928'
  and notice_id= '1928'
  and route_id= '1928';
select agency_id
from m.agency
where agency_id_id= '19384'
  and valid_now=11585;
select agency_id
from m.agency
where agency_id_id= '6547'
  and valid_now=15908;
select COUNT(*)
from dv.notes_message
where user_id='8012'
  and agency_id_id= '8012'
  and notice_id= '8012'
  and route_id= '8012';
select agency_id
from m.agency
where agency_id_id= '3051'
  and valid_now=10562;
select user_id
from m.agency
where valid_now=3941
  and agency_id_id= '593';
select COUNT(*)
from dv.notes_message
where user_id='14932'
  and agency_id_id= '14932'
  and notice_id= '14932'
  and route_id= '14932';
select user_id
from m.agency
where valid_now=16118
  and agency_id_id= '10039';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14804'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3650'
  and valid_now=4064;
select agency_id
from m.agency
where agency_id_id= '17378'
  and valid_now=7176;
select user_id
from m.agency
where valid_now=3416
  and agency_id_id= '5834';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18947'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2339'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1363'
  and valid_now=4355;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5735'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=1520
  and agency_id_id= '4000';
select COUNT(*)
from dv.notes_message
where user_id='1097'
  and agency_id_id= '1097'
  and notice_id= '1097'
  and route_id= '1097';
select COUNT(*)
from dv.notes_message
where user_id='16387'
  and agency_id_id= '16387'
  and notice_id= '16387'
  and route_id= '16387';
select user_id
from m.agency
where valid_now=8392
  and agency_id_id= '6352';
select agency_id
from m.agency
where agency_id_id= '17519'
  and valid_now=10647;
select user_id
from m.agency
where valid_now=7050
  and agency_id_id= '4219';
select COUNT(*)
from dv.notes_message
where user_id='11715'
  and agency_id_id= '11715'
  and notice_id= '11715'
  and route_id= '11715';
select user_id
from m.agency
where valid_now=15632
  and agency_id_id= '2592';
select COUNT(*)
from dv.notes_message
where user_id='6362'
  and agency_id_id= '6362'
  and notice_id= '6362'
  and route_id= '6362';
select agency_id
from m.agency
where agency_id_id= '966'
  and valid_now=15029;
select user_id
from m.agency
where valid_now=8760
  and agency_id_id= '12083';
select user_id
from m.agency
where valid_now=9395
  and agency_id_id= '7625';
select COUNT(*)
from dv.notes_message
where user_id='15777'
  and agency_id_id= '15777'
  and notice_id= '15777'
  and route_id= '15777';
select COUNT(*)
from dv.notes_message
where user_id='1299'
  and agency_id_id= '1299'
  and notice_id= '1299'
  and route_id= '1299';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3955'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7992'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18391
  and agency_id_id= '1855';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5638'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18541
  and agency_id_id= '16850';
select user_id
from m.agency
where valid_now=8667
  and agency_id_id= '12849';
select user_id
from m.agency
where valid_now=16546
  and agency_id_id= '19576';
select user_id
from m.agency
where valid_now=5601
  and agency_id_id= '4950';
select user_id
from m.agency
where valid_now=4966
  and agency_id_id= '3312';
select COUNT(*)
from dv.notes_message
where user_id='12288'
  and agency_id_id= '12288'
  and notice_id= '12288'
  and route_id= '12288';
select agency_id
from m.agency
where agency_id_id= '7172'
  and valid_now=18894;
select agency_id
from m.agency
where agency_id_id= '594'
  and valid_now=18554;
select agency_id
from m.agency
where agency_id_id= '12400'
  and valid_now=5604;
select agency_id
from m.agency
where agency_id_id= '18681'
  and valid_now=15351;
select user_id
from m.agency
where valid_now=8234
  and agency_id_id= '16058';
select user_id
from m.agency
where valid_now=18039
  and agency_id_id= '932';
select agency_id
from m.agency
where agency_id_id= '1938'
  and valid_now=16894;
select user_id
from m.agency
where valid_now=12815
  and agency_id_id= '11766';
select COUNT(*)
from dv.notes_message
where user_id='4079'
  and agency_id_id= '4079'
  and notice_id= '4079'
  and route_id= '4079';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7895'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=19085
  and agency_id_id= '642';
select COUNT(*)
from dv.notes_message
where user_id='13844'
  and agency_id_id= '13844'
  and notice_id= '13844'
  and route_id= '13844';
select user_id
from m.agency
where valid_now=15504
  and agency_id_id= '2721';
select COUNT(*)
from dv.notes_message
where user_id='8129'
  and agency_id_id= '8129'
  and notice_id= '8129'
  and route_id= '8129';
select agency_id
from m.agency
where agency_id_id= '16252'
  and valid_now=8;
select COUNT(*)
from dv.notes_message
where user_id='10792'
  and agency_id_id= '10792'
  and notice_id= '10792'
  and route_id= '10792';
select agency_id
from m.agency
where agency_id_id= '5300'
  and valid_now=16036;
select COUNT(*)
from dv.notes_message
where user_id='5557'
  and agency_id_id= '5557'
  and notice_id= '5557'
  and route_id= '5557';
select agency_id
from m.agency
where agency_id_id= '5376'
  and valid_now=14515;
select user_id
from m.agency
where valid_now=9160
  and agency_id_id= '13321';
select user_id
from m.agency
where valid_now=9266
  and agency_id_id= '15033';
select COUNT(*)
from dv.notes_message
where user_id='19650'
  and agency_id_id= '19650'
  and notice_id= '19650'
  and route_id= '19650';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17219'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10033'
  and valid_now=11500;
select user_id
from m.agency
where valid_now=6904
  and agency_id_id= '1384';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1017'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8173'
  and valid_now=19128;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12063'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16741
  and agency_id_id= '5435';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1814'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2568'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=301
  and agency_id_id= '10914';
select user_id
from m.agency
where valid_now=18518
  and agency_id_id= '13303';
select agency_id
from m.agency
where agency_id_id= '3339'
  and valid_now=10682;
select agency_id
from m.agency
where agency_id_id= '4112'
  and valid_now=687;
select user_id
from m.agency
where valid_now=155
  and agency_id_id= '8294';
select user_id
from m.agency
where valid_now=4425
  and agency_id_id= '11370';
select agency_id
from m.agency
where agency_id_id= '11594'
  and valid_now=5371;
select COUNT(*)
from dv.notes_message
where user_id='10834'
  and agency_id_id= '10834'
  and notice_id= '10834'
  and route_id= '10834';
select user_id
from m.agency
where valid_now=13119
  and agency_id_id= '3542';
select COUNT(*)
from dv.notes_message
where user_id='18860'
  and agency_id_id= '18860'
  and notice_id= '18860'
  and route_id= '18860';
select agency_id
from m.agency
where agency_id_id= '13745'
  and valid_now=15940;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19920'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7656'
  and agency_id_id= '7656'
  and notice_id= '7656'
  and route_id= '7656';
select user_id
from m.agency
where valid_now=9237
  and agency_id_id= '13733';
select COUNT(*)
from dv.notes_message
where user_id='8122'
  and agency_id_id= '8122'
  and notice_id= '8122'
  and route_id= '8122';
select COUNT(*)
from dv.notes_message
where user_id='6566'
  and agency_id_id= '6566'
  and notice_id= '6566'
  and route_id= '6566';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18259'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='3022'
  and agency_id_id= '3022'
  and notice_id= '3022'
  and route_id= '3022';
select COUNT(*)
from dv.notes_message
where user_id='8634'
  and agency_id_id= '8634'
  and notice_id= '8634'
  and route_id= '8634';
select agency_id
from m.agency
where agency_id_id= '4425'
  and valid_now=1896;
select user_id
from m.agency
where valid_now=2238
  and agency_id_id= '5167';
select COUNT(*)
from dv.notes_message
where user_id='6537'
  and agency_id_id= '6537'
  and notice_id= '6537'
  and route_id= '6537';
select agency_id
from m.agency
where agency_id_id= '15873'
  and valid_now=10460;
select agency_id
from m.agency
where agency_id_id= '5185'
  and valid_now=12720;
select user_id
from m.agency
where valid_now=17580
  and agency_id_id= '16544';
select user_id
from m.agency
where valid_now=4162
  and agency_id_id= '92';
select COUNT(*)
from dv.notes_message
where user_id='15876'
  and agency_id_id= '15876'
  and notice_id= '15876'
  and route_id= '15876';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3665'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1902'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '8243'
  and valid_now=9347;
select agency_id
from m.agency
where agency_id_id= '6760'
  and valid_now=97;
select agency_id
from m.agency
where agency_id_id= '8122'
  and valid_now=18465;
select COUNT(*)
from dv.notes_message
where user_id='4756'
  and agency_id_id= '4756'
  and notice_id= '4756'
  and route_id= '4756';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12355'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1836'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4405'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1135'
  and valid_now=4790;
select user_id
from m.agency
where valid_now=611
  and agency_id_id= '18283';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12814'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16163'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17751'
  and agency_id_id= '17751'
  and notice_id= '17751'
  and route_id= '17751';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1191'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12050'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '860'
  and valid_now=17116;
select COUNT(*)
from dv.notes_message
where user_id='18266'
  and agency_id_id= '18266'
  and notice_id= '18266'
  and route_id= '18266';
select COUNT(*)
from dv.notes_message
where user_id='18435'
  and agency_id_id= '18435'
  and notice_id= '18435'
  and route_id= '18435';
select user_id
from m.agency
where valid_now=1291
  and agency_id_id= '9340';
select COUNT(*)
from dv.notes_message
where user_id='12046'
  and agency_id_id= '12046'
  and notice_id= '12046'
  and route_id= '12046';
select agency_id
from m.agency
where agency_id_id= '16885'
  and valid_now=19625;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2863'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11530'
  and valid_now=6157;
select user_id
from m.agency
where valid_now=6899
  and agency_id_id= '3653';
select user_id
from m.agency
where valid_now=14334
  and agency_id_id= '7915';
select COUNT(*)
from dv.notes_message
where user_id='10696'
  and agency_id_id= '10696'
  and notice_id= '10696'
  and route_id= '10696';
select user_id
from m.agency
where valid_now=423
  and agency_id_id= '7491';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '13329'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10155
  and agency_id_id= '12153';
select agency_id
from m.agency
where agency_id_id= '17799'
  and valid_now=9482;
select agency_id
from m.agency
where agency_id_id= '7245'
  and valid_now=19817;
select COUNT(*)
from dv.notes_message
where user_id='14155'
  and agency_id_id= '14155'
  and notice_id= '14155'
  and route_id= '14155';
select COUNT(*)
from dv.notes_message
where user_id='12700'
  and agency_id_id= '12700'
  and notice_id= '12700'
  and route_id= '12700';
select user_id
from m.agency
where valid_now=1254
  and agency_id_id= '19220';
select user_id
from m.agency
where valid_now=662
  and agency_id_id= '8772';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1400'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17434'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='15085'
  and agency_id_id= '15085'
  and notice_id= '15085'
  and route_id= '15085';
select COUNT(*)
from dv.notes_message
where user_id='16881'
  and agency_id_id= '16881'
  and notice_id= '16881'
  and route_id= '16881';
select user_id
from m.agency
where valid_now=12860
  and agency_id_id= '9573';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3868'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=13133
  and agency_id_id= '2998';
select agency_id
from m.agency
where agency_id_id= '11094'
  and valid_now=3133;
select user_id
from m.agency
where valid_now=14318
  and agency_id_id= '3559';
select user_id
from m.agency
where valid_now=13750
  and agency_id_id= '16849';
select COUNT(*)
from dv.notes_message
where user_id='15'
  and agency_id_id= '15'
  and notice_id= '15'
  and route_id= '15';
select COUNT(*)
from dv.notes_message
where user_id='16696'
  and agency_id_id= '16696'
  and notice_id= '16696'
  and route_id= '16696';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5222'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3992'
  and valid_now=16701;
select user_id
from m.agency
where valid_now=2846
  and agency_id_id= '9264';
select user_id
from m.agency
where valid_now=5087
  and agency_id_id= '2372';
select agency_id
from m.agency
where agency_id_id= '6986'
  and valid_now=18854;
select user_id
from m.agency
where valid_now=11423
  and agency_id_id= '8733';
select COUNT(*)
from dv.notes_message
where user_id='8389'
  and agency_id_id= '8389'
  and notice_id= '8389'
  and route_id= '8389';
select agency_id
from m.agency
where agency_id_id= '2726'
  and valid_now=19970;
select COUNT(*)
from dv.notes_message
where user_id='1025'
  and agency_id_id= '1025'
  and notice_id= '1025'
  and route_id= '1025';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12023'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7443
  and agency_id_id= '6296';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18917'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=16522
  and agency_id_id= '13210';
select user_id
from m.agency
where valid_now=12411
  and agency_id_id= '11293';
select agency_id
from m.agency
where agency_id_id= '9369'
  and valid_now=1360;
select user_id
from m.agency
where valid_now=7760
  and agency_id_id= '19382';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18398'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6795'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18261'
  and valid_now=9803;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12344'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18215'
  and valid_now=3257;
select COUNT(*)
from dv.notes_message
where user_id='6866'
  and agency_id_id= '6866'
  and notice_id= '6866'
  and route_id= '6866';
select agency_id
from m.agency
where agency_id_id= '11362'
  and valid_now=6078;
select COUNT(*)
from dv.notes_message
where user_id='1971'
  and agency_id_id= '1971'
  and notice_id= '1971'
  and route_id= '1971';
select COUNT(*)
from dv.notes_message
where user_id='18162'
  and agency_id_id= '18162'
  and notice_id= '18162'
  and route_id= '18162';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17285'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='5272'
  and agency_id_id= '5272'
  and notice_id= '5272'
  and route_id= '5272';
select agency_id
from m.agency
where agency_id_id= '9166'
  and valid_now=11690;
select agency_id
from m.agency
where agency_id_id= '2822'
  and valid_now=16174;
select user_id
from m.agency
where valid_now=9899
  and agency_id_id= '14888';
select COUNT(*)
from dv.notes_message
where user_id='1413'
  and agency_id_id= '1413'
  and notice_id= '1413'
  and route_id= '1413';
select agency_id
from m.agency
where agency_id_id= '4126'
  and valid_now=3420;
select user_id
from m.agency
where valid_now=13282
  and agency_id_id= '14365';
select COUNT(*)
from dv.notes_message
where user_id='9341'
  and agency_id_id= '9341'
  and notice_id= '9341'
  and route_id= '9341';
select COUNT(*)
from dv.notes_message
where user_id='9840'
  and agency_id_id= '9840'
  and notice_id= '9840'
  and route_id= '9840';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4981'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '2561'
  and valid_now=278;
select user_id
from m.agency
where valid_now=7961
  and agency_id_id= '11122';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4848'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '399'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '18476'
  and valid_now=1264;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15462'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '3072'
  and valid_now=15240;
select COUNT(*)
from dv.notes_message
where user_id='3212'
  and agency_id_id= '3212'
  and notice_id= '3212'
  and route_id= '3212';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8841'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6448
  and agency_id_id= '2288';
select agency_id
from m.agency
where agency_id_id= '5889'
  and valid_now=12774;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12648'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16515'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '464'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6673
  and agency_id_id= '13685';
select user_id
from m.agency
where valid_now=4655
  and agency_id_id= '10627';
select COUNT(*)
from dv.notes_message
where user_id='12538'
  and agency_id_id= '12538'
  and notice_id= '12538'
  and route_id= '12538';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1346'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5249'
  and valid_now=2528;
select COUNT(*)
from dv.notes_message
where user_id='15383'
  and agency_id_id= '15383'
  and notice_id= '15383'
  and route_id= '15383';
select agency_id
from m.agency
where agency_id_id= '10183'
  and valid_now=3389;
select agency_id
from m.agency
where agency_id_id= '10953'
  and valid_now=6207;
select user_id
from m.agency
where valid_now=4407
  and agency_id_id= '18571';
select COUNT(*)
from dv.notes_message
where user_id='3551'
  and agency_id_id= '3551'
  and notice_id= '3551'
  and route_id= '3551';
select COUNT(*)
from dv.notes_message
where user_id='14785'
  and agency_id_id= '14785'
  and notice_id= '14785'
  and route_id= '14785';
select COUNT(*)
from dv.notes_message
where user_id='5281'
  and agency_id_id= '5281'
  and notice_id= '5281'
  and route_id= '5281';
select COUNT(*)
from dv.notes_message
where user_id='3426'
  and agency_id_id= '3426'
  and notice_id= '3426'
  and route_id= '3426';
select COUNT(*)
from dv.notes_message
where user_id='6225'
  and agency_id_id= '6225'
  and notice_id= '6225'
  and route_id= '6225';
select COUNT(*)
from dv.notes_message
where user_id='18349'
  and agency_id_id= '18349'
  and notice_id= '18349'
  and route_id= '18349';
select user_id
from m.agency
where valid_now=9000
  and agency_id_id= '16713';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5557'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10293
  and agency_id_id= '1623';
select COUNT(*)
from dv.notes_message
where user_id='19072'
  and agency_id_id= '19072'
  and notice_id= '19072'
  and route_id= '19072';
select COUNT(*)
from dv.notes_message
where user_id='5755'
  and agency_id_id= '5755'
  and notice_id= '5755'
  and route_id= '5755';
select user_id
from m.agency
where valid_now=6945
  and agency_id_id= '5';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1202'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12089'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '11456'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=18562
  and agency_id_id= '9301';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19676'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1011'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=14511
  and agency_id_id= '13908';
select COUNT(*)
from dv.notes_message
where user_id='315'
  and agency_id_id= '315'
  and notice_id= '315'
  and route_id= '315';
select agency_id
from m.agency
where agency_id_id= '2416'
  and valid_now=14910;
select user_id
from m.agency
where valid_now=4404
  and agency_id_id= '6092';
select user_id
from m.agency
where valid_now=15536
  and agency_id_id= '14758';
select user_id
from m.agency
where valid_now=209
  and agency_id_id= '8534';
select user_id
from m.agency
where valid_now=8327
  and agency_id_id= '4866';
select COUNT(*)
from dv.notes_message
where user_id='8652'
  and agency_id_id= '8652'
  and notice_id= '8652'
  and route_id= '8652';
select COUNT(*)
from dv.notes_message
where user_id='8350'
  and agency_id_id= '8350'
  and notice_id= '8350'
  and route_id= '8350';
select agency_id
from m.agency
where agency_id_id= '5401'
  and valid_now=12973;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6732'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '6310'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15942'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=15670
  and agency_id_id= '10719';
select COUNT(*)
from dv.notes_message
where user_id='5893'
  and agency_id_id= '5893'
  and notice_id= '5893'
  and route_id= '5893';
select user_id
from m.agency
where valid_now=4787
  and agency_id_id= '14768';
select COUNT(*)
from dv.notes_message
where user_id='14356'
  and agency_id_id= '14356'
  and notice_id= '14356'
  and route_id= '14356';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5278'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='7095'
  and agency_id_id= '7095'
  and notice_id= '7095'
  and route_id= '7095';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4799'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '5257'
  and valid_now=11085;
select agency_id
from m.agency
where agency_id_id= '9394'
  and valid_now=348;
select user_id
from m.agency
where valid_now=11382
  and agency_id_id= '11856';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2339'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19108'
  and valid_now=3973;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8604'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1406'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8647
  and agency_id_id= '17695';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16091'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '11212'
  and valid_now=19392;
select user_id
from m.agency
where valid_now=3210
  and agency_id_id= '13910';
select user_id
from m.agency
where valid_now=10091
  and agency_id_id= '16819';
select COUNT(*)
from dv.notes_message
where user_id='17134'
  and agency_id_id= '17134'
  and notice_id= '17134'
  and route_id= '17134';
select COUNT(*)
from dv.notes_message
where user_id='10875'
  and agency_id_id= '10875'
  and notice_id= '10875'
  and route_id= '10875';
select user_id
from m.agency
where valid_now=15652
  and agency_id_id= '10467';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3717'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3071
  and agency_id_id= '18836';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2368'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=10703
  and agency_id_id= '2834';
select COUNT(*)
from dv.notes_message
where user_id='3623'
  and agency_id_id= '3623'
  and notice_id= '3623'
  and route_id= '3623';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16818'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6893'
  and agency_id_id= '6893'
  and notice_id= '6893'
  and route_id= '6893';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15905'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13461'
  and valid_now=13291;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1919'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8446
  and agency_id_id= '8188';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4246'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='2468'
  and agency_id_id= '2468'
  and notice_id= '2468'
  and route_id= '2468';
select agency_id
from m.agency
where agency_id_id= '8193'
  and valid_now=6873;
select user_id
from m.agency
where valid_now=1032
  and agency_id_id= '8517';
select user_id
from m.agency
where valid_now=11837
  and agency_id_id= '19614';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9326'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '12694'
  and valid_now=2089;
select user_id
from m.agency
where valid_now=11811
  and agency_id_id= '3131';
select COUNT(*)
from dv.notes_message
where user_id='12928'
  and agency_id_id= '12928'
  and notice_id= '12928'
  and route_id= '12928';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8813'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7862
  and agency_id_id= '9056';
select agency_id
from m.agency
where agency_id_id= '6196'
  and valid_now=834;
select COUNT(*)
from dv.notes_message
where user_id='17229'
  and agency_id_id= '17229'
  and notice_id= '17229'
  and route_id= '17229';
select COUNT(*)
from dv.notes_message
where user_id='1581'
  and agency_id_id= '1581'
  and notice_id= '1581'
  and route_id= '1581';
select agency_id
from m.agency
where agency_id_id= '3829'
  and valid_now=16458;
select user_id
from m.agency
where valid_now=13246
  and agency_id_id= '1367';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8137'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '9859'
  and valid_now=15563;
select COUNT(*)
from dv.notes_message
where user_id='10567'
  and agency_id_id= '10567'
  and notice_id= '10567'
  and route_id= '10567';
select COUNT(*)
from dv.notes_message
where user_id='19698'
  and agency_id_id= '19698'
  and notice_id= '19698'
  and route_id= '19698';
select user_id
from m.agency
where valid_now=18696
  and agency_id_id= '12690';
select agency_id
from m.agency
where agency_id_id= '13809'
  and valid_now=10959;
select agency_id
from m.agency
where agency_id_id= '10721'
  and valid_now=339;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10863'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12901'
  and agency_id_id= '12901'
  and notice_id= '12901'
  and route_id= '12901';
select COUNT(*)
from dv.notes_message
where user_id='17357'
  and agency_id_id= '17357'
  and notice_id= '17357'
  and route_id= '17357';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '17611'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '16199'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1826'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '13280'
  and valid_now=17646;
select COUNT(*)
from dv.notes_message
where user_id='5149'
  and agency_id_id= '5149'
  and notice_id= '5149'
  and route_id= '5149';
select COUNT(*)
from dv.notes_message
where user_id='18981'
  and agency_id_id= '18981'
  and notice_id= '18981'
  and route_id= '18981';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1189'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '10072'
  and valid_now=12623;
select user_id
from m.agency
where valid_now=10451
  and agency_id_id= '18872';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14844'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='19878'
  and agency_id_id= '19878'
  and notice_id= '19878'
  and route_id= '19878';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15570'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=7969
  and agency_id_id= '11054';
select COUNT(*)
from dv.notes_message
where user_id='16222'
  and agency_id_id= '16222'
  and notice_id= '16222'
  and route_id= '16222';
select COUNT(*)
from dv.notes_message
where user_id='17158'
  and agency_id_id= '17158'
  and notice_id= '17158'
  and route_id= '17158';
select user_id
from m.agency
where valid_now=5655
  and agency_id_id= '9342';
select COUNT(*)
from dv.notes_message
where user_id='1861'
  and agency_id_id= '1861'
  and notice_id= '1861'
  and route_id= '1861';
select user_id
from m.agency
where valid_now=651
  and agency_id_id= '4626';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '19411'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19947'
  and valid_now=4098;
select COUNT(*)
from dv.notes_message
where user_id='11483'
  and agency_id_id= '11483'
  and notice_id= '11483'
  and route_id= '11483';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '1191'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=5634
  and agency_id_id= '18868';
select COUNT(*)
from dv.notes_message
where user_id='11288'
  and agency_id_id= '11288'
  and notice_id= '11288'
  and route_id= '11288';
select agency_id
from m.agency
where agency_id_id= '2551'
  and valid_now=10481;
select user_id
from m.agency
where valid_now=13755
  and agency_id_id= '17253';
select COUNT(*)
from dv.notes_message
where user_id='10578'
  and agency_id_id= '10578'
  and notice_id= '10578'
  and route_id= '10578';
select COUNT(*)
from dv.notes_message
where user_id='11875'
  and agency_id_id= '11875'
  and notice_id= '11875'
  and route_id= '11875';
select agency_id
from m.agency
where agency_id_id= '11430'
  and valid_now=9816;
select user_id
from m.agency
where valid_now=18534
  and agency_id_id= '15695';
select COUNT(*)
from dv.notes_message
where user_id='6215'
  and agency_id_id= '6215'
  and notice_id= '6215'
  and route_id= '6215';
select COUNT(*)
from dv.notes_message
where user_id='11894'
  and agency_id_id= '11894'
  and notice_id= '11894'
  and route_id= '11894';
select agency_id
from m.agency
where agency_id_id= '17964'
  and valid_now=17736;
select agency_id
from m.agency
where agency_id_id= '6623'
  and valid_now=15001;
select COUNT(*)
from dv.notes_message
where user_id='5291'
  and agency_id_id= '5291'
  and notice_id= '5291'
  and route_id= '5291';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5782'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '5452'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=314
  and agency_id_id= '5687';
select user_id
from m.agency
where valid_now=17558
  and agency_id_id= '10879';
select COUNT(*)
from dv.notes_message
where user_id='5633'
  and agency_id_id= '5633'
  and notice_id= '5633'
  and route_id= '5633';
select agency_id
from m.agency
where agency_id_id= '16997'
  and valid_now=13828;
select user_id
from m.agency
where valid_now=19905
  and agency_id_id= '15606';
select COUNT(*)
from dv.notes_message
where user_id='5135'
  and agency_id_id= '5135'
  and notice_id= '5135'
  and route_id= '5135';
select agency_id
from m.agency
where agency_id_id= '2933'
  and valid_now=9260;
select user_id
from m.agency
where valid_now=17012
  and agency_id_id= '17333';
select agency_id
from m.agency
where agency_id_id= '17459'
  and valid_now=1379;
select user_id
from m.agency
where valid_now=977
  and agency_id_id= '542';
select user_id
from m.agency
where valid_now=17742
  and agency_id_id= '6061';
select user_id
from m.agency
where valid_now=7558
  and agency_id_id= '6175';
select COUNT(*)
from dv.notes_message
where user_id='1944'
  and agency_id_id= '1944'
  and notice_id= '1944'
  and route_id= '1944';
select COUNT(*)
from dv.notes_message
where user_id='12358'
  and agency_id_id= '12358'
  and notice_id= '12358'
  and route_id= '12358';
select a.agency_timezone
from m.agency a
where a.agency_id = '5283';
select agency_id
from m.agency
where agency_id_id= '19543'
  and valid_now=4592;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '3580'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '19435'
  and valid_now=6556;
select COUNT(*)
from dv.notes_message
where user_id='7507'
  and agency_id_id= '7507'
  and notice_id= '7507'
  and route_id= '7507';
select agency_id
from m.agency
where agency_id_id= '2413'
  and valid_now=2906;
select COUNT(*)
from dv.notes_message
where user_id='15523'
  and agency_id_id= '15523'
  and notice_id= '15523'
  and route_id= '15523';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4316'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17718'
  and agency_id_id= '17718'
  and notice_id= '17718'
  and route_id= '17718';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '4790'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='16963'
  and agency_id_id= '16963'
  and notice_id= '16963'
  and route_id= '16963';
select COUNT(*)
from dv.notes_message
where user_id='14319'
  and agency_id_id= '14319'
  and notice_id= '14319'
  and route_id= '14319';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '12719'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '15334'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='381'
  and agency_id_id= '381'
  and notice_id= '381'
  and route_id= '381';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10640'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='17990'
  and agency_id_id= '17990'
  and notice_id= '17990'
  and route_id= '17990';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9594'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=6897
  and agency_id_id= '11074';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '14387'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '814'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='6762'
  and agency_id_id= '6762'
  and notice_id= '6762'
  and route_id= '6762';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18060'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '7046'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select agency_id
from m.agency
where agency_id_id= '1750'
  and valid_now=6444;
select agency_id
from m.agency
where agency_id_id= '15719'
  and valid_now=7337;
select agency_id
from m.agency
where agency_id_id= '8778'
  and valid_now=2760;
select user_id
from m.agency
where valid_now=15301
  and agency_id_id= '15177';
select COUNT(*)
from dv.notes_message
where user_id='13994'
  and agency_id_id= '13994'
  and notice_id= '13994'
  and route_id= '13994';
select COUNT(*)
from dv.notes_message
where user_id='11344'
  and agency_id_id= '11344'
  and notice_id= '11344'
  and route_id= '11344';
select COUNT(*)
from dv.notes_message
where user_id='18099'
  and agency_id_id= '18099'
  and notice_id= '18099'
  and route_id= '18099';
select COUNT(*)
from dv.notes_message
where user_id='5041'
  and agency_id_id= '5041'
  and notice_id= '5041'
  and route_id= '5041';
select user_id
from m.agency
where valid_now=10370
  and agency_id_id= '14680';
select COUNT(*)
from dv.notes_message
where user_id='12473'
  and agency_id_id= '12473'
  and notice_id= '12473'
  and route_id= '12473';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '8463'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=12898
  and agency_id_id= '19423';
select agency_id
from m.agency
where agency_id_id= '12929'
  and valid_now=15093;
select COUNT(*)
from dv.notes_message
where user_id='10181'
  and agency_id_id= '10181'
  and notice_id= '10181'
  and route_id= '10181';
select user_id
from m.agency
where valid_now=4014
  and agency_id_id= '10182';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10132'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select COUNT(*)
from dv.notes_message
where user_id='12842'
  and agency_id_id= '12842'
  and notice_id= '12842'
  and route_id= '12842';
select user_id
from m.agency
where valid_now=16386
  and agency_id_id= '14347';
select user_id
from m.agency
where valid_now=13825
  and agency_id_id= '12084';
select user_id
from m.agency
where valid_now=5319
  and agency_id_id= '15018';
select COUNT(*)
from dv.notes_message
where user_id='15040'
  and agency_id_id= '15040'
  and notice_id= '15040'
  and route_id= '15040';
select a.agency_timezone
from m.agency a
where a.agency_id = '18992';
select user_id
from m.agency
where valid_now=3423
  and agency_id_id= '5408';
select a.agency_timezone
from m.agency a
where a.agency_id = '9227';
select user_id
from m.agency
where valid_now=9586
  and agency_id_id= '10391';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '2721'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '9662'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=3292
  and agency_id_id= '2702';
select COUNT(*)
from dv.notes_message
where user_id='8148'
  and agency_id_id= '8148'
  and notice_id= '8148'
  and route_id= '8148';
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '18178'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select distinct ea.agency_id,
                c.start_date
from m.agency a,
     m.agency ea,
     m.calendar c
where a.agency_id = '10380'
  and ea.agency_id_id = a.agency_id_id
  and ea.agency_id = c.agency_id
  and (
         (select extract(epoch
                         from c.start_date)*1)) <= 1
  and (
         (select extract(epoch
                         from c.end_date +1)*1)) >= 1;
select user_id
from m.agency
where valid_now=8312
  and agency_id_id= '2623';
select COUNT(*)
from dv.notes_message
where user_id='1583'
  and agency_id_id= '1583'
  and notice_id= '1583'
  and route_id= '1583';
select user_id
from m.agency
where valid_now=3195
  and agency_id_id= '14818';
select COUNT(*)
from dv.notes_message
where user_id='4000'
  and agency_id_id= '4000'
  and notice_id= '4000'
  and route_id= '4000';
select COUNT(*)
from dv.notes_message
where user_id='17343'
  and agency_id_id= '17343'
  and notice_id= '17343'
  and route_id= '17343';
select a.agency_timezone
from m.agency a
where a.agency_id = '19188';
select COUNT(*)
from dv.notes_message
where user_id='11496'
  and agency_id_id= '11496'
  and notice_id= '11496'
  and route_id= '11496';
select a.agency_timezone
from m.agency a
where a.agency_id = '3830';
select a.agency_timezone
from m.agency a
where a.agency_id = '5255';
select a.agency_timezone
from m.agency a
where a.agency_id = '13979';
select user_id
from m.agency
where valid_now=19611
  and agency_id_id= '7934';
select COUNT(*)
from dv.notes_message
where user_id='11222'
  and agency_id_id= '11222'
  and notice_id= '11222'
  and route_id= '11222';
select user_id
from m.agency
where valid_now=7923
  and agency_id_id= '6071';
select COUNT(*)
from dv.notes_message
where user_id='17704'
  and agency_id_id= '17704'
  and notice_id= '17704'
  and route_id= '17704';
select COUNT(*)
from dv.notes_message
where user_id='10348'
  and agency_id_id= '10348'
  and notice_id= '10348'
  and route_id= '10348';
