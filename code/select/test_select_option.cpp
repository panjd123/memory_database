/**
 * @file test_select_option.cpp
 * @author ruichenhan (hanruichen@ruc.edu.cn)
 * @brief Selection operator implementation
 * @version 0.1
 * @date 2023-10-03
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/metadata.h"
#include "../include/gendata_util.hpp"
#include "../include/statistical_analysis_util.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
/**
 * @brief the shared dynamic selection vector implementations: non-braching code
 * @param n the num of tuples
 * @param sd_sv the shared dynamic selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_dsel_shared_val_non_branching(idx n,
                                           std::vector<T> &sd_sv,
                                           const T *col1,
                                           T *val2)
{
  idx i = 0, j = 0;
  idx current_idx = 0;
  if (sd_sv.size() == 0)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      j += (col1[i] <= val2[0]);
      if (j > sd_sv.size())
        sd_sv.emplace_back(i);
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {

      j += (col1[sd_sv[i]] <= *val2);
      if (current_idx < j)
      {
        sd_sv[current_idx] = sd_sv[i];
        current_idx++;
      }
    }
  }
  return j;
}

/**
 * @brief the shared fixed selection vector implementations: non-braching code
 * @param n the num of tuples
 * @param sf_sv the shared fixed selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_shared_val_non_branching(idx n,
                                           idx *sf_sv,
                                           const T *col1,
                                           T *val2,
                                           idx current_size)
{
  idx i = 0, j = 0;
  idx current_idx = 0;
  if (current_size == 0)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      sf_sv[j] = i;
      j += (col1[i] <= val2[0]);
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {

      j += (col1[sf_sv[i]] <= *val2);
      if (current_idx < j)
      {
        sf_sv[current_idx] = sf_sv[i];
        current_idx++;
      }
    }
  }
  return j;
}
/**
 * @brief the shared bitmap implementations: non-braching code
 * @param n the num of tuples
 * @param s_bitmap the shared bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
void sel_lt_T_bmp_shared_val_non_branching(idx n,
                                           std::vector<bool> &s_bitmap,
                                           const T *col1,
                                           T *val2,
                                           bool firstflag)
{
  idx i = 0;
  if (firstflag)
  {
    for (i = 0; i < n; i++)
      s_bitmap[i] = (col1[i] <= val2[0]);
  }
  else
  {
    for (i = 0; i < n; i++)
      s_bitmap[i] = (col1[i] <= val2[0]) && s_bitmap[i];
  }
  return;
}
/**
 * @brief the independent dynamic selection vector implementations: non-braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
void sel_lt_T_dsel_independent_val_non_branching(idx n,
                                                 std::vector<T> &res,
                                                 const T *col1,
                                                 T *val2,
                                                 std::vector<T> &id_sv)
{
  idx i = 0, j = 0;
  if (id_sv.size() == 0)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      j += (col1[i] <= val2[0]);
      if (j > res.size())
        res.emplace_back(i);
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {

      j += (col1[id_sv[i]] <= *val2);
      if (j > res.size())
        res.emplace_back(id_sv[i]);
    }
  }
  return;
}
/**
 * @brief the independent fixed selection vector implementations: non-braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param if_sv the independent fixed selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_independent_val_non_branching(idx n,
                                                int *res,
                                                const T *col1,
                                                T *val2,
                                                int *if_sv)
{
  idx i = 0, j = 0;
  if (if_sv == NULL)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      res[j] = i;
      j += (col1[i] <= val2[0]);
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {
      res[j] = if_sv[i];
      j += (col1[if_sv[i]] <= *val2);
    }
  }
  return j;
}

/**
 * @brief the independent bitmap implementations: non-braching code
 * @param n the num of tuples
 * @param i_bitmap the independent bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_independent_val_non_branching(idx n,
                                                std::vector<bool> &i_bitmap,
                                                const T *col1,
                                                T *val2)
{
  idx i = 0;

  for (i = 0; i < n; i++)
  {

    i_bitmap[i] = (col1[i] <= val2[0]);
  }

  return;
}
/**
 * @brief the shared dynamic selection vector implementations: braching code
 * @param n the num of tuples
 * @param sd_sv the shared dynamic selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_dsel_shared_val_branching(idx n,
                                       std::vector<T> &sd_sv,
                                       const T *col1,
                                       T *val2)
{
  idx i = 0, j = 0;
  if (sd_sv.size() == 0)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[i] <= *val2)
      {
        j++;
        sd_sv.emplace_back(i);
      }
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[sd_sv[i]] <= *val2)
      {
        sd_sv[j++] = sd_sv[i];
      }
    }
  }
  return j;
}
/**
 * @brief the shared bitmap implementations: braching code
 * @param n the num of tuples
 * @param s_bitmap the shared bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_shared_val_branching(idx n,
                                       std::vector<bool> &s_bitmap,
                                       const T *col1,
                                       T *val2,
                                       bool firstflag)
{
  idx i = 0;
  if (firstflag)
  {
    for (i = 0; i < n; i++)
    {
      if (col1[i] <= *val2)
      {
        s_bitmap[i] = 1;
      }
      else
      {
        s_bitmap[i] = 0;
      }
    }
  }
  else
  {
    for (i = 0; i < n; i++)
    {
      if (s_bitmap[i] && (col1[i] <= *val2))
      {
        s_bitmap[i] = 1;
      }
      else
      {
        s_bitmap[i] = 0;
      }
    }
  }
  return;
}

/**
 * @brief the shared fixed selection vector implementations: braching code
 * @param n the num of tuples
 * @param sf_sv the shared fixed selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @param current_size the length of the selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_shared_val_branching(idx n,
                                       idx *sf_sv,
                                       const T *col1,
                                       T *val2,
                                       idx current_size)
{
  idx i = 0, j = 0;
  if (current_size == 0)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[i] <= *val2)
      {
        sf_sv[j++] = i;
      }
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[sf_sv[i]] <= *val2)
      {
        sf_sv[j++] = sf_sv[i];
      }
    }
  }
  return j;
}

/**
 * @brief the independent dynamic selection vector implementations:braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
void sel_lt_T_dsel_independent_val_branching(idx n,
                                             std::vector<T> &res,
                                             const T *col1,
                                             T *val2,
                                             std::vector<T> &id_sv)
{
  idx i = 0;
  if (id_sv.size() == 0)
  {
    for (i = 0; i < n; i++)
    {
      if (col1[i] <= *val2)
      {
        res.emplace_back(i);
      }
    }
  }
  else
  {
    for (i = 0; i < n; i++)
    {
      if (col1[id_sv[i]] <= *val2)
      {
        res.emplace_back(id_sv[i]);
      }
    }
  }
  return;
}

/**
 * @brief the independent fixed selection vector implementations:braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param if_sv the independent fixed selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_independent_val_branching(idx n,
                                            int *res,
                                            const T *col1,
                                            T *val2,
                                            int *if_sv)
{
  idx i = 0, j = 0;
  if (if_sv == NULL)
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[i] <= *val2)
        res[j++] = i;
    }
  }
  else
  {
    for (i = 0, j = 0; i < n; i++)
    {
      if (col1[if_sv[i]] <= *val2)
        res[j++] = if_sv[i];
    }
  }
  return j;
}
/**
 * @brief the independent bitmap implementations:braching code
 * @param n the num of tuples
 * @param i_bitmap the independent bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_independent_val_branching(idx n,
                                            std::vector<bool> &i_bitmap,
                                            const T *col1,
                                            T *val2)
{
  idx i = 0;

  for (i = 0; i < n; i++)
  {
    if (col1[i] <= *val2)
      i_bitmap[i] = 1;
    else
      i_bitmap[i] = 0;
  }

  return;
}
/**
 * @brief perform selection operator using Row-wise query processing model
 * @param condition determine the selection rate
 * @param Ra Filter column Ra
 * @param Rb Filter column Rb
 * @param Rc Filter column Rc
 * @param Rd Measure column Rd
 * @return int the count of selection result
 */
idx selalgo_rowwise(idx condition, const idx &size_R,
                    const T *Ra,
                    const T *Rb,
                    const T *Rc,
                    const T *Rd)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  for (i = 0; i != result_size; ++i)
  {
    if (Ra[i] <= condition && Rb[i] <= condition && Rc[i] <= condition)
    {
      count += Rd[i];
    }
  }
  return count;
}
/**
 * @brief perform selection operator using Culomn-wise query processing model with shared dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra Filter column Ra
 * @param Rb Filter column Rb
 * @param Rc Filter column Rc
 * @param Rd Measure column Rd
 * @param sd_sv the shared dynamic selection vector
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           std::vector<T> &sd_sv,
                           const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    current_size_ra = sel_lt_T_dsel_shared_val_non_branching(result_size, sd_sv, Ra, &condition);
    current_size_rb = sel_lt_T_dsel_shared_val_non_branching(current_size_ra, sd_sv, Rb, &condition);
    current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
    current_size_rb = sel_lt_T_dsel_shared_val_branching(current_size_ra, sd_sv, Rb, &condition);
    current_size_rc = sel_lt_T_dsel_shared_val_branching(current_size_rb, sd_sv, Rc, &condition);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
    current_size_rb = sel_lt_T_dsel_shared_val_branching(current_size_ra, sd_sv, Rb, &condition);
    current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  else
  {
    current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
    current_size_rb = sel_lt_T_dsel_shared_val_non_branching(current_size_ra, sd_sv, Rb, &condition);
    current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  return count;
}
/**
 * @brief perform selection operator using Culomn-wise query processing model with shared fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra Filter column Ra
 * @param Rb Filter column Rb
 * @param Rc Filter column Rc
 * @param Rd Measure column Rd
 * @param sf_sv the shared fixed selection vector
 * @return int the count of selection result
 */
idx selalgo_cwm_fsv_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           idx *sf_sv,
                           const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    current_size_ra = sel_lt_T_fsel_shared_val_non_branching(result_size, sf_sv, Ra, &condition, 0);
    current_size_rb = sel_lt_T_fsel_shared_val_non_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
    current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
    current_size_rb = sel_lt_T_fsel_shared_val_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
    current_size_rc = sel_lt_T_fsel_shared_val_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
    current_size_rb = sel_lt_T_fsel_shared_val_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
    current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else
  {
    current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
    current_size_rb = sel_lt_T_fsel_shared_val_non_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
    current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  return count;
}
/**
 * @brief perform selection operator using Culomn-wise query processing model with shared bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra Filter column Ra
 * @param Rb Filter column Rb
 * @param Rc Filter column Rc
 * @param Rd Measure column Rd
 * @param s_bitmap the shared bitmap
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           std::vector<bool> &s_bitmap,
                           const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Ra, &condition, true);
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rb, &condition, false);
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rb, &condition, false);
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rc, &condition, false);
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rb, &condition, false);
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  else
  {
    sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rb, &condition, false);
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  return count;
}

/**
 * @brief perform selection operator using Culomn-wise query processing model with independent dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_independent(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<T> &d_sv_ra,
                                std::vector<T> &d_sv_rb,
                                std::vector<T> &d_sv_rc,
                                const Selalgo_Branch &selalgo_branch)
{
  idx count = 0, count1 = 0;
  idx i;
  idx result_size = size_R;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    sel_lt_T_dsel_independent_val_non_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    for (i = 0; i < d_sv_rc.size(); i++)
    {
      count += Rd[d_sv_rc[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {

    sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    for (i = 0; i < d_sv_rc.size(); i++)
    {
      count += Rd[d_sv_rc[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    for (i = 0; i < d_sv_rc.size(); i++)
    {
      count += Rd[d_sv_rc[i]];
    }
  }
  else
  {
    sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    for (i = 0; i < d_sv_rc.size(); i++)
    {
      count += Rd[d_sv_rc[i]];
    }
  }
  return count;
}

/**
 * @brief perform select using Culomn-wise query processing model with independent fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_fsv_independent(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                int *f_sv_ra,
                                int *f_sv_rb,
                                int *f_sv_rc,
                                const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    current_size_ra = sel_lt_T_fsel_independent_val_non_branching(result_size, f_sv_ra, Ra, &condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[f_sv_rc[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[f_sv_rc[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[f_sv_rc[i]];
    }
  }
  else
  {
    current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[f_sv_rc[i]];
    }
  }
  return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with independent bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_independent(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<bool> &bitmap_Ra,
                                std::vector<bool> &bitmap_Rb,
                                std::vector<bool> &bitmap_Rc,
                                std::vector<bool> &bitmap,
                                const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
  {
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Ra, Ra, &condition);
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rb, Rb, &condition);
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
    for (i = 0; i != result_size; ++i)
    {

      bitmap[i] = (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i]);
    }
    for (i = 0; i < result_size; i++)
    {
      if (bitmap[i])
        count += Rd[i];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rb, Rb, &condition);
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rc, Rc, &condition);
    for (i = 0; i != result_size; ++i)
    {
      if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
        bitmap[i] = 1;
      else
        bitmap[i] = 0;
    }
    for (i = 0; i < result_size; i++)
    {
      if (bitmap[i])
        count += Rd[i];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rb, Rb, &condition);
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
    for (i = 0; i != result_size; ++i)
    {
      if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
        bitmap[i] = 1;
      else
        bitmap[i] = 0;
    }
    for (i = 0; i < result_size; i++)
    {
      if (bitmap[i])
        count += Rd[i];
    }
  }
  else
  {
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rb, Rb, &condition);
    sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
    for (i = 0; i != result_size; ++i)
    {
      if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
        bitmap[i] = 1;
      else
        bitmap[i] = 0;
    }
    for (i = 0; i < result_size; i++)
    {
      if (bitmap[i])
        count += Rd[i];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sd_sv the shared dynamic selection vector
 * @return int the count of selection result
 */
idx casetest_combined_cwm_dsv_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     std::vector<T> &sd_sv,
                                     const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
      {
        sd_sv.emplace_back(i);
      }
    }
    for (i = 0; i < sd_sv.size(); i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        sd_sv.emplace_back(i);
      }
    }
    current_size_rc = sel_lt_T_dsel_shared_val_non_branching(sd_sv.size(), sd_sv, Rc, &condition);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sd_sv[i]];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with independent selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_dsv_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          std::vector<T> &d_sv_ra,
                                          std::vector<T> &d_sv_rb,
                                          std::vector<T> &d_sv_rc,
                                          const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;

  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        d_sv_rb.emplace_back(i);
      }
    }
    sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    for (i = 0; i < d_sv_rc.size(); i++)
    {
      count += Rd[d_sv_rc[i]];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with shared fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sf_sv the shared fixed selection vector
 * @return int the count of selection result
 */
idx casetest_combined_cwm_fsv_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     idx *sf_sv,
                                     const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
      {
        sf_sv[current_size_rc++] = i;
      }
    }
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        sf_sv[current_size_rc] = i;
        current_size_rc += (Rc[i] <= condition);
      }
    }
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE)
  {
    for (i = 0; i < result_size; i++)
    {
      if (Ra[i] <= condition)
      {
        sf_sv[current_size_rc] = i;
        current_size_rc += ((Rb[i] <= condition) && (Rc[i] <= condition));
      }
    }
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  else
  {
    for (i = 0; i < result_size; i++)
    {
      sf_sv[current_size_rc] = i;
      current_size_rc += ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition));
    }
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[sf_sv[i]];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with independent fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_fsv_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          int *f_sv_ra,
                                          int *f_sv_rb,
                                          int *f_sv_rc,
                                          const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  idx current_size_ra = 0;
  idx current_size_rb = 0;
  idx current_size_rc = 0;
  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        f_sv_rb[current_size_rb++] = i;
      }
    }
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
    for (i = 0; i < current_size_rc; i++)
    {
      count += Rd[f_sv_rb[i]];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with shared bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param s_bitmap the shared bitmap
 * @return int the count of selection result
 */
idx casetest_combined_cwm_bmp_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     std::vector<bool> &s_bitmap,
                                     const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
      {
        s_bitmap[i] = 1;
      }
      else
      {
        s_bitmap[i] = 0;
      }
    }
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        s_bitmap[i] = 1;
      }
      else
      {
        s_bitmap[i] = 0;
      }
    }
    sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
    for (i = 0; i < result_size; i++)
    {
      if (s_bitmap[i])
        count += Rd[i];
    }
  }
  return count;
}
/**
 * @brief case test using combined column-wise model with independent bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_bmp_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          std::vector<bool> &bitmap_Ra,
                                          std::vector<bool> &bitmap_Rb,
                                          std::vector<bool> &bitmap_Rc,
                                          std::vector<bool> &bitmap,
                                          const Selalgo_Branch &selalgo_branch)
{
  idx count = 0;
  idx i;
  idx result_size = size_R;
  if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
  {
    for (i = 0; i < result_size; i++)
    {
      if ((Ra[i] <= condition) && (Rb[i] <= condition))
      {
        bitmap_Rb[i] = 1;
      }
      else
      {
        bitmap_Rb[i] = 0;
      }
    }
    sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rc, Rc, &condition);
    for (i = 0; i != result_size; ++i)
    {
      if (bitmap_Rb[i] && bitmap_Rc[i])
        bitmap[i] = 1;
      else
        bitmap[i] = 0;
    }
    for (i = 0; i < result_size; i++)
    {
      if (bitmap[i])
        count += Rd[i];
    }
  }
  return count;
}
/**
 * @brief row-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_rowwise_model(const idx &size_R,
                                const T *Ra, const T *Rb,
                                const T *Rc, const T *Rd,
                                const std::vector<idx> &conditions,
                                std::ofstream &selalgo_model_timefile,
                                std::ofstream &selalgo_model_lsr_timefile,
                                bool is_lsr)
{
  std::cout << ">>> Start test using row-wise model" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    gettimeofday(&start, NULL);
    count = selalgo_rowwise(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                 << "\t"
                                 << "Row-wise query model"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Row-wise query model"
                             << "\t"
                             << "BRANCH_ONE_TWO_THREE"
                             << "\t"
                             << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
    }
  }
}
/**
 * @brief cloumn-wise query processing model with different dynamic vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_dsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)
{
  /*the shared dynamic selection vector*/
  std::cout << ">>> Start selection operator test using column-wise model with the shared dynamic vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3)  * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    std::vector<int> sd_sv;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = selalgo_cwm_dsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sd_sv, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] /1000 , 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the shared dynamic selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the shared dynamic selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the shared dynamic selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE" 
                         << "\t"
                         <<"Column-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the shared dynamic selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the shared dynamic selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
  /*the independent dynamic selection vector*/
  std::cout << ">>> Start selection operator test using column-wise model with the independent dynamic vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3)  * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    std::vector<int> d_sv_ra;
    std::vector<int> d_sv_rb;
    std::vector<int> d_sv_rc;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = selalgo_cwm_dsv_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, d_sv_ra, d_sv_rb, d_sv_rc, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3)  * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the independent dynamic selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the independent dynamic selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the independent dynamic selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the independent dynamic selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}
/**
 * @brief vector-wise query processing model with different dynamic vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_dsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)
{
  /*the shared dynamic selection vector*/
  std::cout << ">>> Start selection operator test using vector-wise model with the shared dynamic vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    std::vector<int> sd_sv;
    sd_sv.reserve(size_v);
    idx vec_num = DATA_NUM / size_v;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_dsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                      sd_sv, selalgo_branch);
      sd_sv.clear();
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the shared dynamic selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the shared dynamic selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t"
                       << "the shared dynamic selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Vector-wise model with the shared dynamic selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
  /*the independent dynamic selection vector*/
  std::cout << ">>> Start selection operator test using vector-wise model with the independent dynamic vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    std::vector<int> d_sv_ra;
    std::vector<int> d_sv_rb;
    std::vector<int> d_sv_rc;
    d_sv_ra.reserve(size_v);
    d_sv_rb.reserve(size_v);
    d_sv_rc.reserve(size_v);
    idx vec_num = DATA_NUM / size_v;
    //std::cout << vec_num << std::endl;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_dsv_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                           d_sv_ra, d_sv_rb, d_sv_rc, selalgo_branch);
      d_sv_ra.clear();
      d_sv_rb.clear();
      d_sv_rc.clear();
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the independent dynamic selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the independent dynamic selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t"
                       << "the independent dynamic selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Vector-wise model with the independent dynamic selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}

/**
 * @brief cloumn-wise query processing model with different fixed vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_fsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &casestudy_timefile,
                          std::ofstream &casestudy_lsr_timefile,
                          bool is_lsr)
{
  /*the shared fixed selection vector*/
  std::cout << ">>> Start selection operator test using column-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    const idx vector_size = size_R;
    int *sf_sv = new int[vector_size];
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = selalgo_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the shared fixed selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
      casestudy_lsr_timefile << "multipass processing mode"
                             << "\t"
                             << "Column-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "multipass processing mode with Column-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the shared fixed selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the shared fixed selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Column-wise model with the shared fixed selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the shared fixed selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the shared fixed selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the shared fixed selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
      casestudy_timefile << "multipass processing mode"
                         << "\t"
                         << "Column-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "multipass processing mode with Column-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }

  /*the independent fixed selection vector*/
  std::cout << ">>> Start selection operator test using column-wise model with the independent fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    const idx vector_size = size_R;
    int *f_sv_ra = new int[vector_size];
    int *f_sv_rb = new int[vector_size];
    int *f_sv_rc = new int[vector_size];
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = selalgo_cwm_fsv_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, f_sv_ra, f_sv_rb, f_sv_rc, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the independent fixed selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the independent fixed selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the independent fixed selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Column-wise model with the independent fixed selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the independent fixed selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the independent fixed selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the independent fixed selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}
/**
 * @brief vector-wise query processing model with different fixed vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_fsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &casestudy_timefile,
                          std::ofstream &casestudy_lsr_timefile,
                          bool is_lsr)
{
  /*the shared fixed selection vector*/
  std::cout << ">>> Start selection operator test using vector-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3)  * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;

    int *sf_sv = new int[size_v];
    idx vec_num = DATA_NUM / size_v;

    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3)  * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the shared fixed selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
      casestudy_lsr_timefile << "multipass processing mode"
                             << "\t"
                             << "Vector-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "multipass processing mode with Vector-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the shared fixed selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t"
                       << "the shared fixed selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Vector-wise model with the shared fixed selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
      casestudy_timefile << "multipass processing mode"
                         << "\t"
                         << "Vector-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "multipass processing mode with Vector-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }
  /*the independent fixed selection vector*/
  std::cout << ">>> Start selection operator test using vector-wise model with the independent fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    const idx vector_size = size_R;
    int *f_sv_ra = new int[size_v];
    int *f_sv_rb = new int[size_v];
    int *f_sv_rc = new int[size_v];
    idx vec_num = DATA_NUM / size_v;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_fsv_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, f_sv_ra, f_sv_rb, f_sv_rc, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the independent fixed selection vector"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the independent fixed selection vector"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t"
                       << "the independent fixed selection vector"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Vector-wise model with the independent fixed selection vector and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}
/**
 * @brief cloumn-wise query processing model with different bitmap for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_bmp(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)

{
  /*the shared bitmap*/
  std::cout << ">>> Start selection operator test using column-wise model with the shared bitmap" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    std::vector<bool> bitmap;
    bitmap.reserve(DATA_NUM);
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    gettimeofday(&start, NULL);
    count = selalgo_cwm_bmp_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, bitmap, selalgo_branch);

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    // time_results.emplace_back(ms);
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the shared bitmap"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the shared bitmap"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the shared bitmap"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Column-wise model with the shared bitmap and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the shared bitmap and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the shared bitmap and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the shared bitmap and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
  /*the independent bitmap*/
  std::cout << ">>> Start selection operator test using column-wise model with the independent bitmap" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    std::vector<bool> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
    bitmap_Ra.reserve(DATA_NUM);
    bitmap_Rb.reserve(DATA_NUM);
    bitmap_Rc.reserve(DATA_NUM);
    bitmap.reserve(DATA_NUM);
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    gettimeofday(&start, NULL);
    count = selalgo_cwm_bmp_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap, selalgo_branch);

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    // time_results.emplace_back(ms);
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Column-wise query model with the independent bitmap"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Column-wise query model with the independent bitmap"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Column-wise query model"
                       << "\t"
                       << "the independent bitmap"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                         << "\t"
                         << "Column-wise model with the independent bitmap and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "BRANCH_ONE_TWO"
                         << "\t"
                         << "Column-wise model with the independent bitmap and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "BRANCH_ONE"
                         << "\t"
                         << "Column-wise model with the independent bitmap and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Column-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "NON_BRANCH"
                         << "\t"
                         << "Column-wise model with the independent bitmap and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Column-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}
/**
 * @brief vector-wise query processing model with different bitmap for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_bmp(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)

{
  std::cout << ">>> Start selection operator test using vector-wise model with the shared bitmap" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    std::vector<bool> bitmap;
    bitmap.reserve(size_v);
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_bmp_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, bitmap, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    // time_results.emplace_back(ms);
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the shared bitmap"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the shared bitmap"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "Vector-wise model with the shared bitmap and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
  std::cout << ">>> Start selection operator test using vector-wise model with the independent bitmap" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    std::vector<bool> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
    bitmap_Ra.reserve(size_v);
    bitmap_Rb.reserve(size_v);
    bitmap_Rc.reserve(size_v);
    bitmap.reserve(size_v);
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_bmp_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                           bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    // time_results.emplace_back(ms);
    if (is_lsr)
    {
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                   << "\t";
        break;
      case 1:
        selalgo_model_lsr_timefile << "BRANCH_ONE_TWO"
                                   << "\t";
      case 2:
        selalgo_model_lsr_timefile << "BRANCH_ONE"
                                   << "\t";
      case 3:
        selalgo_model_lsr_timefile << "NON_BRANCH"
                                   << "\t";
      default:
        break;
      }
      selalgo_model_lsr_timefile << "Vector-wise query model with the independent bitmap"
                                 << "\t"
                                 << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                 << "\t"
                                 << ms << std::endl;
    }
    else
    {
      selalgo_model_timefile << "Vector-wise query model with the independent bitmap"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_model_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        selalgo_model_timefile << "BRANCH_ONE_TWO"
                               << "\t";
      case 2:
        selalgo_model_timefile << "BRANCH_ONE"
                               << "\t";
      case 3:
        selalgo_model_timefile << "NON_BRANCH"
                               << "\t";
      default:
        break;
      }
      selalgo_model_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
      selalgo_timefile << "Vector-wise query model"
                       << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE_TWO_THREE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                         << "\t";
        break;
      case 1:
        selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE_TWO"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                         << "\t";
      case 2:
        selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE"
                         << "\t";
        selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                         << "\t";
      case 3:
        selalgo_timefile << "Vector-wise model with the independent bitmap and NON_BRANCH"
                         << "\t";
        selalgo_timefile << "Vector-wise model with NON_BRANCH"
                         << "\t";
      default:
        break;
      }
      selalgo_timefile << 0.1 * (select_idx + 1)
                       << "\t"
                       << ms << std::endl;
    }
  }
}
/**
 * @brief column-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_columnwise_model(const idx &size_R,
                                   const T *Ra, const T *Rb,
                                   const T *Rc, const T *Rd,
                                   const std::vector<idx> &conditions,
                                   const Selalgo_Branch &selalgo_branch,
                                   std::ofstream &selalgo_model_timefile,
                                   std::ofstream &selalgo_model_lsr_timefile,
                                   std::ofstream &selalgo_timefile,
                                   std::ofstream &casestudy_timefile,
                                   std::ofstream &casestudy_lsr_timefile,
                                   bool is_lsr)
{
  /*column-wise query processing model with dynamic selection vector*/
  test_selalgo_cwm_dsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
  /*column-wise query processing model with fixed selection vector*/
  test_selalgo_cwm_fsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  /*column-wise query processing model with bitmap*/
  test_selalgo_cwm_bmp(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
}
/**
 * @brief vector-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vectorwise_model(const idx &size_R,
                                   const T *Ra, const T *Rb,
                                   const T *Rc, const T *Rd,
                                   const std::vector<idx> &conditions,
                                   const Selalgo_Branch &selalgo_branch,
                                   std::ofstream &selalgo_model_timefile,
                                   std::ofstream &selalgo_model_lsr_timefile,
                                   std::ofstream &selalgo_timefile,
                                   std::ofstream &casestudy_timefile,
                                   std::ofstream &casestudy_lsr_timefile,
                                   bool is_lsr)
{
  /*vector-wise query processing model with dynamic selection vector*/
  test_selalgo_vwm_dsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
  /*vector-wise query processing model with fixed selection vector*/
  test_selalgo_vwm_fsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  /*vector-wise query processing model with bitmap*/
  test_selalgo_vwm_bmp(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
}
/**
 * @brief combined column-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_combined_columnwise_model(const idx &size_R,
                                         const T *Ra, const T *Rb,
                                         const T *Rc, const T *Rd,
                                         const std::vector<idx> &conditions,
                                         const Selalgo_Branch &selalgo_branch,
                                         std::ofstream &casestudy_timefile,
                                         std::ofstream &casestudy_lsr_timefile,
                                         bool is_lsr)
{
  /*the shared fixed selection vector*/
  std::cout << ">>> Start case test using combined column-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;

    const idx vector_size = size_R;
    int *sf_sv = new int[vector_size];
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = casetest_combined_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      casestudy_lsr_timefile << "combined processing mode"
                             << "\t"
                             << "Column-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "combined processing mode with Column-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      casestudy_timefile << "combined processing mode"
                         << "\t"
                         << "Column-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "combined processing mode with Column-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }
}
/**
 * @brief multipass column-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_multipass_columnwise_model(const idx &size_R,
                                          const T *Ra, const T *Rb,
                                          const T *Rc, const T *Rd,
                                          const std::vector<idx> &conditions,
                                          const Selalgo_Branch &selalgo_branch,
                                          std::ofstream &casestudy_timefile,
                                          std::ofstream &casestudy_lsr_timefile,
                                          bool is_lsr)
{

  /*the shared fixed selection vector*/
  std::cout << ">>> Start case test using multipass column-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;

    const idx vector_size = size_R;
    int *sf_sv = new int[vector_size];
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    count = selalgo_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      casestudy_lsr_timefile << "multipass processing mode"
                             << "\t"
                             << "Column-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "multipass processing mode with Column-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      casestudy_timefile << "multipass processing mode"
                         << "\t"
                         << "Column-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "multipass processing mode with Column-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }
}
/**
 * @brief combined vector-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_combined_vectorwise_model(const idx &size_R,
                                         const T *Ra, const T *Rb,
                                         const T *Rc, const T *Rd,
                                         const std::vector<idx> &conditions,
                                         const Selalgo_Branch &selalgo_branch,
                                         std::ofstream &casestudy_timefile,
                                         std::ofstream &casestudy_lsr_timefile,
                                         bool is_lsr)
{

  /*the shared fixed selection vector*/
  std::cout << ">>> Start case test using combined vector-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    int *sf_sv = new int[size_v];
    idx vec_num = DATA_NUM / size_v;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += casetest_combined_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      casestudy_lsr_timefile << "combined processing mode"
                             << "\t"
                             << "Vector-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "combined processing mode with Vector-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      casestudy_timefile << "combined processing mode"
                         << "\t"
                         << "Vector-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "combined processing mode with Vector-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }
}
/**
 * @brief multipass vector-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_multipass_vectorwise_model(const idx &size_R,
                                          const T *Ra, const T *Rb,
                                          const T *Rc, const T *Rd,
                                          const std::vector<idx> &conditions,
                                          const Selalgo_Branch &selalgo_branch,
                                          std::ofstream &casestudy_timefile,
                                          std::ofstream &casestudy_lsr_timefile,
                                          bool is_lsr)
{

  /*the shared fixed selection vector*/
  std::cout << ">>> Start case test using multipass vector-wise model with the shared fixed vector" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
  {
    if (is_lsr)
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    else
      std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " <<  pow((double)conditions[select_idx] / 1000, 3) * 100 << "%" << std::endl;
    idx count = 0;
    timeval start, end;
    int *sf_sv = new int[size_v];
    idx vec_num = DATA_NUM / size_v;
    // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += selalgo_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
    }
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    if (is_lsr)
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    else
      std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 1000, 3) * 100 << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    if (is_lsr)
    {
      casestudy_lsr_timefile << "multipass processing mode"
                             << "\t"
                             << "Vector-wise query model"
                             << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                               << "\t";
        break;
      case 1:
        casestudy_lsr_timefile << "BRANCH_ONE_TWO"
                               << "\t";
        break;
      case 2:
        casestudy_lsr_timefile << "BRANCH_ONE"
                               << "\t";
        break;
      case 3:
        casestudy_lsr_timefile << "NON_BRANCH"
                               << "\t";
        break;
      default:
        break;
      }
      casestudy_lsr_timefile << "multipass processing mode with Vector-wise query model"
                             << "\t"
                             << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                             << "\t"
                             << ms << std::endl;
    }
    else
    {
      casestudy_timefile << "multipass processing mode"
                         << "\t"
                         << "Vector-wise query model"
                         << "\t";
      switch ((int)selalgo_branch)
      {
      case 0:
        casestudy_timefile << "BRANCH_ONE_TWO_THREE"
                           << "\t";
        break;
      case 1:
        casestudy_timefile << "BRANCH_ONE_TWO"
                           << "\t";
        break;
      case 2:
        casestudy_timefile << "BRANCH_ONE"
                           << "\t";
        break;
      case 3:
        casestudy_timefile << "NON_BRANCH"
                           << "\t";
        break;
      default:
        break;
      }
      casestudy_timefile << "multipass processing mode with Vector-wise query model"
                         << "\t"
                         << 0.1 * (select_idx + 1)
                         << "\t"
                         << ms << std::endl;
    }
  }
}
/**
 * @brief selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo(const idx &size_R,
                  const T *Ra, const T *Rb,
                  const T *Rc, const T *Rd,
                  const std::vector<idx> &conditions,
                  std::ofstream &selalgo_model_timefile,
                  std::ofstream &selalgo_model_lsr_timefile,
                  std::ofstream &selalgo_timefile,
                  std::ofstream &casestudy_timefile,
                  std::ofstream &casestudy_lsr_timefile,
                  bool is_lsr)
{
  /*1.Row-wise query processing model*/
  test_selalgo_rowwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_model_timefile, selalgo_model_lsr_timefile, is_lsr);

  for (const auto branch : SELALGO_BRANCH)
  {
    /*2. Column-wise query processing model*/
    test_selalgo_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    /*3. Vector-wise query processing model*/
    test_selalgo_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  }
}
/**
 * @brief comparison test for selection algorithm cases
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case(const idx &size_R,
               const T *Ra, const T *Rb,
               const T *Rc, const T *Rd,
               const std::vector<idx> &conditions,
               std::ofstream &casestudy_timefile,
               std::ofstream &casestudy_lsr_timefile,
               bool is_lsr)
{
  for (const auto branch : CASE_COMBINED_BRANCH)
  {
    /*1. combined column-wise processing model*/
    test_case_combined_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    /*2. combined vector-wise processing model*/
    test_case_combined_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  }
  for (const auto branch : CASE_MULTIPASS_BRANCH)
  {
    /*3. multi-pass column-wise query processing model*/
    test_case_multipass_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    /*4. multi-pass vector-wise query processing model*/
    test_case_multipass_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  }
}
int main(int argc, char **argv)
{
  bool is_lsr = false;
  static int is_lsr_flag;
  int opt;
  static struct option long_options[] =
      {
          /* These options set a flag. */
          {"is_lsr", no_argument, &is_lsr_flag, 1},
          {0, 0, 0, 0}};
  const char *optstring = "ab:nr:";
  int option_index = 0;
  opt = getopt_long(argc, argv, optstring, long_options,
                    &option_index);
  is_lsr = is_lsr_flag;
  std::ofstream selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile;
  if (is_lsr)
  {
    selalgo_model_lsr_timefile.open(SELALGO_MODEL_LSR_TIME_FILE, std::ios::out | std::ios::trunc);
    casestudy_lsr_timefile.open(CASESTUDY_LSR_TIME_FILE, std::ios::out | std::ios::trunc);
  }
  else
  {
    selalgo_model_timefile.open(SELALGO_MODEL_TIME_FILE, std::ios::out | std::ios::trunc);
    selalgo_timefile.open(SELALGO_TIME_FILE, std::ios::out | std::ios::trunc);
    casestudy_timefile.open(CASESTUDY_TIME_FILE, std::ios::out | std::ios::trunc);
  }
  if (is_lsr)
  {
    selalgo_model_lsr_timefile << "Branching type"
                               << "\t"
                               << "Query processing model with different Intermediate Result Type"
                               << "\t"
                               << "Lg(Selection rate)"
                               << "\t"
                               << "Runtimes(ms)" << std::endl;
    casestudy_lsr_timefile << "Processing mode"
                           << "\t"
                           << "Query model"
                           << "\t"
                           << "Branching type"
                           << "\t"
                           << "Processing model with different Query model"
                           << "\t"
                           << "Lg(Selection rate)"
                           << "\t"
                           << "Runtimes(ms)" << std::endl;
  }
  else
  {
    selalgo_model_timefile << "Query processing model with different Intermediate Result Type"
                           << "\t"
                           << "Branching type "
                           << "\t"
                           << "Selection rate"
                           << "\t"
                           << "Runtimes(ms)" << std::endl;

    selalgo_timefile << "Query processing model"
                     << "\t"
                     << "Intermediate Result Type"
                     << "\t"
                     << "Branching type"
                     << "\t"
                     << "Query processing model with different Intermediate Result Type and Branching type"
                     << "\t"
                     << "Query processing model with different Branching type"
                     << "\t"
                     << "Selection rate"
                     << "\t"
                     << "Runtimes(ms)" << std::endl;
    casestudy_timefile << "Processing mode"
                       << "\t"
                       << "Query model"
                       << "\t"
                       << "Branching type"
                       << "\t"
                       << "Processing model with different Query model"
                       << "\t"
                       << "Selection rate"
                       << "\t"
                       << "Runtimes(ms)" << std::endl;
  }

  T *Ra = new T[DATA_NUM];
  T *Rb = new T[DATA_NUM];
  T *Rc = new T[DATA_NUM];
  T *Rd = new T[DATA_NUM];
  std::vector<int> conditions;

  gen_data(DATA_NUM, Ra, Rb, Rc, Rd, is_lsr);
  gen_conditions(conditions, is_lsr);
  /*Selection algorithms for branching and non-branching  implementations and test*/
  test_selalgo(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  /*Case study*/
  test_case(DATA_NUM, Ra, Rb, Rc, Rd, conditions, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
  delete[] Ra;
  delete[] Rb;
  delete[] Rc;
  delete[] Rd;
  conditions.clear();
  selalgo_model_timefile.close();
  selalgo_model_lsr_timefile.close();
  selalgo_timefile.close();
  casestudy_timefile.close();
  casestudy_lsr_timefile.close();
}
