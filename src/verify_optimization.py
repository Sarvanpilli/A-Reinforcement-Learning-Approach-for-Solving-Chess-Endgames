from BaseParams import BoardPossitionParams
import time

if __name__ == '__main__':
    bp = BoardPossitionParams()
    # Create a small subset of params
    all_params = bp.get_all_params()
    params = all_params[:200] 
    
    print(f"Testing with {len(params)} states...")
    start = time.time()
    nxt = bp.get_possible_nxt_prms(params)
    end = time.time()
    
    print(f"Processed {len(params)} states in {end - start:.4f} seconds")
    print(f"Result size: {len(nxt)}")
