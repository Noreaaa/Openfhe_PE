#include "../../src/init/encrypted_region.hpp"
#include <iostream>
#include <random>

static void check(bool ok) { if(!ok)throw std::runtime_error("client region mapping mismatch"); }
int main() {
    try {
        auto ten=MapEncryptedConvRegion({107,116,107,116},224,224,7,7,2,3);
        auto eighty=MapEncryptedConvRegion({72,151,72,151},224,224,7,7,2,3);
        check(ten.hs==52 && ten.he==59 && ten.ws==52 && ten.we==59);
        check(eighty.hs==35 && eighty.he==77 && eighty.ws==35 && eighty.we==77);
        auto second=MapEncryptedConvRegion(eighty,112,112,3,3,2,1);
        check(second.hs==17 && second.he==39);
        auto full=MapEncryptedConvRegion({0,223,0,223},224,224,7,7,2,3);
        check(full.hs==0 && full.he==111 && full.ws==0 && full.we==111);
        auto edge=MapEncryptedConvRegion({0,9,214,223},224,224,7,7,2,3);
        check(edge.hs==0 && edge.he==6 && edge.ws==106 && edge.we==111);
        std::mt19937 generator(123);
        // Compare mapped bounds to explicit receptive-field pixels, including edges and rectangles.
        for(int trial=0;trial<500;++trial) {
            int height=8+generator()%12,width=8+generator()%12,kernel=3+2*(generator()%3),stride=1+generator()%2,pad=kernel/2;
            int hs=generator()%height,he=hs+generator()%(height-hs),ws=generator()%width,we=ws+generator()%(width-ws);
            auto mapped=MapEncryptedConvRegion({hs,he,ws,we},height,width,kernel,kernel,stride,pad);
            for(int oh=0;oh<(height+2*pad-kernel)/stride+1;++oh)
                for(int ow=0;ow<(width+2*pad-kernel)/stride+1;++ow) {
                    bool touched=false;
                    for(int kh=0;kh<kernel;++kh)for(int kw=0;kw<kernel;++kw) {
                        int h=oh*stride+kh-pad,w=ow*stride+kw-pad;
                        touched|=h>=hs && h<=he && w>=ws && w<=we;
                    }
                    check(touched==(oh>=mapped.hs && oh<=mapped.he && ow>=mapped.ws && ow<=mapped.we));
                }
        }
        auto point=MapEncryptedConvRegion({1,1,2,2},8,8,1,1,1,0);
        check(point.hs==1 && point.he==1 && point.ws==2 && point.we==2);
        bool missed=false;try {MapEncryptedConvRegion({1,1,1,1},8,8,1,1,2,0);}catch(const std::invalid_argument&){missed=true;}
        check(missed);
        bool rejected=false;try {MapEncryptedConvRegion({-1,3,0,2},8,8,3,3,2,1);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected);
        std::cout<<"Client region mapping: center, edges, whole image and 500 brute-force checks passed.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
