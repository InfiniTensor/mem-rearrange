#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]

use itertools::izip;
use ndarray_layout::ArrayLayout;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{cmp::Ordering, ptr::copy_nonoverlapping};

pub extern crate ndarray_layout;

/// 存储重排任务对象。
// Layout: | unit | dst offset | src offset | count | idx strides | dst strides | src strides |
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Rearranging(Box<[isize]>);

/// 重排方案异常。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum SchemeError {
    /// 输入输出布局形状不一致。
    ShapeMismatch,
    /// 输出布局中含有广播维度，导致写规约。
    DimReduce,
}

impl Rearranging {
    /// 从输出布局、输入布局和单元规模构造重排方案。
    ///
    /// 单元规模 `unit` 是数组中单个元素的字节数。
    pub fn new<const M: usize, const N: usize>(
        dst: &ArrayLayout<M>,
        src: &ArrayLayout<N>,
        unit: usize,
    ) -> Result<Self, SchemeError> {
        // # 检查基本属性
        let ndim = dst.ndim();
        if src.ndim() != ndim {
            return Err(SchemeError::ShapeMismatch);
        }
        // # 输入形状
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Dim {
            len: usize,
            dst: isize,
            src: isize,
        }
        let mut dims = Vec::with_capacity(ndim);
        for (&dd, &ds, &sd, &ss) in izip!(dst.shape(), src.shape(), dst.strides(), src.strides()) {
            if dd != ds {
                return Err(SchemeError::ShapeMismatch);
            }
            // 剔除初始的 1 长维度
            if dd != 1 {
                if sd == 0 {
                    return Err(SchemeError::DimReduce);
                }
                dims.push(Dim {
                    len: dd,
                    dst: sd,
                    src: ss,
                })
            }
        }
        // # 排序
        dims.sort_unstable_by(|a, b| {
            use Ordering::Equal as Eq;
            match a.dst.abs().cmp(&b.dst.abs()) {
                Eq => match a.src.abs().cmp(&b.src.abs()) {
                    Eq => a.len.cmp(&b.len),
                    ord => ord.reverse(),
                },
                ord => ord.reverse(),
            }
        });
        // # 合并连续维度
        let mut unit = unit as isize;
        let mut ndim = dims.len();
        // ## 合并末尾连续维度到 unit
        for dim in dims.iter_mut().rev() {
            if dim.dst == unit && dim.src == unit {
                unit *= dim.len as isize;
                ndim -= 1
            } else {
                break;
            }
        }
        dims.truncate(ndim);
        // ## 合并任意连续维度
        for i in (1..dims.len()).rev() {
            let (head, tail) = dims.split_at_mut(i);
            let f = &mut head[i - 1]; // f for front
            let b = &mut tail[0]; // b for back
            let len = b.len as isize;
            if b.dst * len == f.dst && b.src * len == f.src {
                *f = Dim {
                    len: b.len * f.len,
                    dst: b.dst,
                    src: b.src,
                };
                *b = Dim {
                    len: 1,
                    dst: 0,
                    src: 0,
                };
                ndim -= 1
            }
        }
        // # 合并空间
        let mut ans = Self(vec![0isize; 4 + ndim * 3].into_boxed_slice());
        ans.0[0] = unit as _;
        ans.0[1] = dst.offset();
        ans.0[2] = src.offset();
        let layout = &mut ans.0[3..];
        layout[ndim] = 1;
        for (i, Dim { len, dst, src }) in dims.into_iter().filter(|d| d.len != 1).enumerate() {
            layout[i] = len as _;
            layout[i + 1 + ndim] = dst;
            layout[i + 1 + ndim * 2] = src;
        }
        for i in (1..=ndim).rev() {
            layout[i - 1] *= layout[i]
        }
        Ok(ans)
    }

    /// 从候选值中选择一个能整除当前单元规模的值作为新的单元规模
    pub fn distribute_unit(&self, candidates: impl IntoIterator<Item = usize>) -> Option<Self> {
        let unit = candidates.into_iter().find(|n| self.unit() % n == 0)?;
        if unit == self.unit() {
            return Some(self.clone());
        }

        let ndim = self.ndim();
        let mut layout = vec![0isize; 4 + (ndim + 1) * 3].into_boxed_slice();
        layout[0] = unit as _;
        layout[1] = self.dst_offset();
        layout[2] = self.src_offset();

        let (_, tail) = layout.split_at_mut(3);
        let (idx, tail) = tail.split_at_mut(ndim + 2);
        let (dst, src) = tail.split_at_mut(ndim + 1);

        let (_, tail) = self.0.split_at(3);
        let (idx_, tail) = tail.split_at(ndim + 1);
        let (dst_, src_) = tail.split_at(ndim);

        idx[ndim + 1] = 1;
        let extra = (self.unit() / unit) as isize;
        for (new, old) in izip!(idx, idx_) {
            *new = *old * extra;
        }

        fn copy_value(new: &mut [isize], old: &[isize], unit: usize) {
            let [head @ .., tail] = new else {
                unreachable!()
            };
            head.copy_from_slice(old);
            *tail = unit as _;
        }
        copy_value(dst, dst_, unit);
        copy_value(src, src_, unit);

        Some(Self(layout))
    }

    /// 执行方案维数。
    #[inline]
    pub fn ndim(&self) -> usize {
        (self.0.len() - 4) / 3
    }

    /// 读写单元规模。
    #[inline]
    pub fn unit(&self) -> usize {
        self.0[0] as _
    }

    /// 输出基址偏移。
    #[inline]
    pub fn dst_offset(&self) -> isize {
        self.0[1]
    }

    /// 输入基址偏移。
    #[inline]
    pub fn src_offset(&self) -> isize {
        self.0[2]
    }

    /// 读写单元数量。
    #[inline]
    pub fn count(&self) -> usize {
        self.0[3] as _
    }

    /// 索引步长。
    #[inline]
    pub fn idx_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[4..][..ndim]
    }

    /// 输出数据步长。
    #[inline]
    pub fn dst_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[4 + ndim..][..ndim]
    }

    /// 输入数据步长。
    #[inline]
    pub fn src_strides(&self) -> &[isize] {
        let ndim = self.ndim();
        &self.0[4 + ndim * 2..][..ndim]
    }

    /// 计算方案涉及的形状。
    pub fn shape(&self) -> impl Iterator<Item = usize> + '_ {
        let ndim = self.ndim();
        self.0[3..][..ndim + 1]
            .windows(2)
            .map(|pair| (pair[0] / pair[1]) as usize)
    }

    /// 执行存储重排。
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid pointers and must able to access with the scheme.
    pub unsafe fn launch(&self, dst: *mut u8, src: *const u8) {
        let dst = unsafe { dst.byte_offset(self.dst_offset()) };
        let src = unsafe { src.byte_offset(self.src_offset()) };
        match self.count() {
            1 => unsafe { copy_nonoverlapping(src, dst, self.unit()) },
            count => {
                let dst = dst as isize;
                let src = src as isize;
                let idx_strides = self.idx_strides();
                let dst_strides = self.dst_strides();
                let src_strides = self.src_strides();
                (0..count as isize).into_par_iter().for_each(|mut rem| {
                    let mut dst = dst;
                    let mut src = src;
                    for (i, &s) in idx_strides.iter().enumerate() {
                        let k = rem / s;
                        dst += k * dst_strides[i];
                        src += k * src_strides[i];
                        rem %= s
                    }
                    unsafe { copy_nonoverlapping::<u8>(src as _, dst as _, self.unit()) }
                })
            }
        }
    }
}

#[test]
fn test_scheme() {
    let shape = [4, 3, 2, 1, 2, 3, 4];
    let dst = [288, 96, 48, 48, 24, 8, 2];
    let src = [576, 192, 96, 48, 8, 16, 2];
    let dst = ArrayLayout::<7>::new(&shape, &dst, 0);
    let src = ArrayLayout::<7>::new(&shape, &src, 0);
    let scheme = Rearranging::new(&dst, &src, 2).unwrap();
    assert_eq!(scheme.ndim(), 3);
    assert_eq!(scheme.dst_offset(), 0);
    assert_eq!(scheme.src_offset(), 0);
    assert_eq!(scheme.unit(), 8);
    assert_eq!(scheme.count(), 24 * 2 * 3);
    assert_eq!(scheme.idx_strides(), [6, 3, 1]);
    assert_eq!(scheme.dst_strides(), [48, 24, 8]);
    assert_eq!(scheme.src_strides(), [96, 8, 16]);
    assert_eq!(scheme.shape().collect::<Vec<_>>(), [24, 2, 3]);
}

#[test]
fn test_distribute_unit() {
    // 创建一个测试用的重排方案
    let shape = [4, 3, 2];
    let dst = [24, 8, 2];
    let src = [48, 8, 16];
    let dst = ArrayLayout::<3>::new(&shape, &dst, 0);
    let src = ArrayLayout::<3>::new(&shape, &src, 0);
    let scheme = Rearranging::new(&dst, &src, 2).unwrap();

    // 测试1: 使用相同的单元大小
    let candidates = vec![2];
    let new_scheme = scheme.distribute_unit(candidates).unwrap();
    assert_eq!(new_scheme.unit(), 2);
    assert_eq!(new_scheme.count(), scheme.count());
    assert_eq!(new_scheme.idx_strides(), scheme.idx_strides());
    assert_eq!(new_scheme.dst_strides(), scheme.dst_strides());
    assert_eq!(new_scheme.src_strides(), scheme.src_strides());

    // 测试2: 使用更小的单元大小
    let candidates = vec![1];
    let new_scheme = scheme.distribute_unit(candidates).unwrap();
    assert_eq!(new_scheme.unit(), 1);
    assert_eq!(new_scheme.count(), scheme.count() * 2);
    assert_eq!(
        new_scheme
            .idx_strides()
            .iter()
            .take(scheme.idx_strides().len())
            .map(|&x| x / 2)
            .collect::<Vec<_>>(),
        scheme.idx_strides()
    );
    assert_eq!(
        new_scheme
            .dst_strides()
            .iter()
            .take(scheme.idx_strides().len())
            .cloned()
            .collect::<Vec<_>>(),
        scheme.dst_strides()
    );
    assert_eq!(
        new_scheme
            .src_strides()
            .iter()
            .take(scheme.idx_strides().len())
            .cloned()
            .collect::<Vec<_>>(),
        scheme.src_strides()
    );

    // 测试3: 使用多个候选值
    let candidates = vec![4, 2, 1];
    let new_scheme = scheme.distribute_unit(candidates).unwrap();
    assert_eq!(new_scheme.unit(), 2); // 应该选择第一个能整除的值
}
