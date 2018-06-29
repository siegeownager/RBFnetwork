import java.util.Random;

public class Node {

	public double bias;
	public double weight;
	double randMinimum = -1.0;
	double randMaximum = 1.0;
	
	public Node(){
		Random randomNum = new Random();
		double num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
		bias = num;
		
		num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
		weight = num;
	}
}
