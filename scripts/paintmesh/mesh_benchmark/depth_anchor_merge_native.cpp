// Exact scheduling acceleration for depth_surface_anchors.merge_anchors.
// Input sorting, normal normalization and scale levels are frozen in Python.
// No fast-math, no changed threshold, no sample cap, no representative motion.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct Mapping {
    int fd=-1;void* data=MAP_FAILED;size_t bytes=0;
    Mapping(const std::string& path,size_t expected) {
        fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("Cannot open "+path);
        struct stat s{};if(fstat(fd,&s)||static_cast<size_t>(s.st_size)!=expected)throw std::runtime_error("Wrong input byte length "+path);
        bytes=expected;
        if(bytes){data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)throw std::runtime_error("mmap failed "+path);}
    }
    ~Mapping(){if(data!=MAP_FAILED)munmap(data,bytes);if(fd>=0)close(fd);}
    Mapping(const Mapping&)=delete;Mapping& operator=(const Mapping&)=delete;
};

struct Key {
    int32_t level;int64_t x,y,z;
    bool operator==(const Key& o)const{return level==o.level&&x==o.x&&y==o.y&&z==o.z;}
};
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Hash {size_t operator()(const Key& k)const{return mix(static_cast<uint64_t>(k.level))^mix(static_cast<uint64_t>(k.x)+0x9e3779b97f4a7c15ULL)^mix(static_cast<uint64_t>(k.y)+0x243f6a8885a308d3ULL)^mix(static_cast<uint64_t>(k.z)+0x13198a2e03707344ULL);}};

static int64_t coord(double p,double cell){
    double q=std::floor(p/cell);
    if(!std::isfinite(q)||q<=static_cast<double>(std::numeric_limits<int64_t>::min()+1)||q>=static_cast<double>(std::numeric_limits<int64_t>::max()-1))throw std::runtime_error("Hash coordinate overflows int64");
    return static_cast<int64_t>(q);
}
template<typename T>static void save(const std::string& path,const std::vector<T>& values){
    std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot write "+path);
    if(values.size())f.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T)));
    if(!f)throw std::runtime_error("Output write failed "+path);
}

int main(int argc,char** argv){
    try {
        if(argc!=9)throw std::runtime_error("Usage: binary folder count max_level distance plane normal_cos scale_ratio radians_to_degrees");
        const std::string folder=argv[1];const uint64_t count=std::stoull(argv[2]);
        const int max_level=std::stoi(argv[3]);
        const double distance_limit=std::stod(argv[4]),plane_limit=std::stod(argv[5]),normal_cos=std::stod(argv[6]),scale_ratio=std::stod(argv[7]);
        const double radians_to_degrees=std::stod(argv[8]);
        if(!count||max_level<0)throw std::runtime_error("Nonempty input and nonnegative levels required");
        Mapping pm(folder+"/positions.raw",count*3*sizeof(double)),nm(folder+"/normals.raw",count*3*sizeof(double));
        Mapping sm(folder+"/scale.raw",count*sizeof(double)),om(folder+"/order.raw",count*sizeof(int64_t));
        Mapping lm(folder+"/levels.raw",count*sizeof(int32_t)),cm(folder+"/cells.raw",(max_level+2)*sizeof(double));
        const double* positions=static_cast<const double*>(pm.data);const double* normals=static_cast<const double*>(nm.data);
        const double* scales=static_cast<const double*>(sm.data);const int64_t* order=static_cast<const int64_t*>(om.data);
        const int32_t* levels=static_cast<const int32_t*>(lm.data);const double* widths=static_cast<const double*>(cm.data);
        std::unordered_map<Key,std::vector<int64_t>,Hash> buckets;
        // Reserve hash buckets only; all real representatives are still inserted.
        buckets.reserve(static_cast<size_t>(count));
        std::vector<int64_t> kept;kept.reserve(count);std::vector<int32_t> multiplicity;multiplicity.reserve(count);
        uint64_t candidate_pairs=0;double merged_distance=0,merged_plane=0,merged_angle=0;
        for(uint64_t index_in_order=0;index_in_order<count;++index_in_order){
            const int64_t index=order[index_in_order];if(index<0||static_cast<uint64_t>(index)>=count)throw std::runtime_error("Invalid sorted source index");
            const double* p=positions+3*index;const double* n=normals+3*index;const double scale=scales[index];const int level=levels[index];
            bool found=false;std::tuple<double,int64_t,double,double> best;
            for(int other_level=level-1;other_level<=level+1;++other_level){
                if(other_level<0)continue;
                if(other_level>max_level+1)throw std::runtime_error("Invalid level");
                const double cell=widths[other_level];const int64_t cx=coord(p[0],cell),cy=coord(p[1],cell),cz=coord(p[2],cell);
                for(int ox=-1;ox<=1;++ox)for(int oy=-1;oy<=1;++oy)for(int oz=-1;oz<=1;++oz){
                    auto bin=buckets.find(Key{other_level,cx+ox,cy+oy,cz+oz});if(bin==buckets.end())continue;
                    for(int64_t slot:bin->second){
                        const int64_t representative=kept[slot];const double rs=scales[representative],small=std::min(scale,rs);
                        if(std::max(scale,rs)>scale_ratio*small)continue;
                        ++candidate_pairs;
                        const double* rp=positions+3*representative;const double* rn=normals+3*representative;
                        const double dx=p[0]-rp[0],dy=p[1]-rp[1],dz=p[2]-rp[2];
                        const double distance=std::sqrt((dx*dx+dy*dy)+dz*dz);
                        if(distance>distance_limit*small)continue;
                        double dot=(n[0]*rn[0]+n[1]*rn[1])+n[2]*rn[2];dot=std::max(-1.,std::min(1.,dot));
                        if(dot<normal_cos)continue;
                        const double own_plane=(dx*n[0]+dy*n[1])+dz*n[2];
                        const double rep_plane=(dx*rn[0]+dy*rn[1])+dz*rn[2];
                        const double plane=std::max(std::abs(own_plane),std::abs(rep_plane));
                        if(plane>plane_limit*small)continue;
                        const double angle=std::acos(dot)*radians_to_degrees;
                        auto match=std::make_tuple(distance,slot,plane,angle);
                        if(!found||match<best){best=match;found=true;}
                    }
                }
            }
            if(found){
                const auto slot=std::get<1>(best);if(multiplicity[slot]==std::numeric_limits<int32_t>::max())throw std::runtime_error("Observation count overflows original int32 schema");
                ++multiplicity[slot];const double small=std::min(scale,scales[kept[slot]]);
                merged_distance=std::max(merged_distance,std::get<0>(best)/small);
                merged_plane=std::max(merged_plane,std::get<2>(best)/small);merged_angle=std::max(merged_angle,std::get<3>(best));
            }else{
                const int64_t slot=static_cast<int64_t>(kept.size());kept.push_back(index);multiplicity.push_back(1);
                const double cell=widths[level];buckets[Key{level,coord(p[0],cell),coord(p[1],cell),coord(p[2],cell)}].push_back(slot);
            }
            if(index_in_order%1000000==0)std::cerr<<"NATIVE_ANCHOR_MERGE "<<index_in_order<<"/"<<count<<" representatives "<<kept.size()<<std::endl;
        }
        save(folder+"/kept.raw",kept);save(folder+"/multiplicity.raw",multiplicity);
        std::ofstream stats(folder+"/stats.json");stats<<std::setprecision(17)
            <<"{\"input_anchors\":"<<count<<",\"output_anchors\":"<<kept.size()<<",\"candidate_pairs_checked\":"<<candidate_pairs
            <<",\"maximum_merged_distance_source_pixels\":"<<merged_distance
            <<",\"maximum_merged_point_plane_source_pixels\":"<<merged_plane
            <<",\"maximum_merged_normal_angle_degrees\":"<<merged_angle<<"}\n";
        if(!stats)throw std::runtime_error("Statistics write failed");
        return 0;
    }catch(const std::exception& error){std::cerr<<"NATIVE_MERGE_ERROR "<<error.what()<<std::endl;return 1;}
}
