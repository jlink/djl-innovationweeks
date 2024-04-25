package tutorial01;

import org.junit.jupiter.api.*;

import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;

/**
 * See http://docs.djl.ai/docs/demos/jupyter/tutorial/01_create_your_first_network.html
 */
class Tutorial01Tests {

	@Test
	void runTutorial() {
		Application application = Application.CV.IMAGE_CLASSIFICATION;
		long inputSize = 28*28;
		long outputSize = 10;
		SequentialBlock block = new SequentialBlock();
		block.add(Blocks.batchFlattenBlock(inputSize));
		block.add(Linear.builder().setUnits(128).build());
		block.add(Activation::relu);
		block.add(Linear.builder().setUnits(64).build());
		block.add(Activation::relu);
		block.add(Linear.builder().setUnits(outputSize).build());

		System.out.println(block);

	}

}
