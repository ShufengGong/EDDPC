package eddpc.dpclu;


/**
 * @author gongsf
 *
 */

import org.apache.hadoop.util.ProgramDriver;

public class Driver {

	public static void main(String[] args)
	{
		ProgramDriver pgd = new ProgramDriver();
		try {
			pgd.addClass("prejob", PreJob.class, "preprocess of EDDPC");
			pgd.addClass("fjob", FirstJob.class, "the first job for compute rho");
			pgd.addClass("sjob", SecondJob.class, "second job for refine delta");
			pgd.driver(args);
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}
}
