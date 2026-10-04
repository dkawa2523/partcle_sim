/** Executes the one-seed, 20 us validation slice of the M3-C2A COMSOL runner. */
public final class RunM3C2StochasticRunnerValidation {
  private RunM3C2StochasticRunnerValidation() {}

  public static void main(String[] args) throws Exception {
    RunM3C2StochasticCampaign.runValidation();
  }
}
